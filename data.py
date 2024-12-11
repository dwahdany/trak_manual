import json
import math

import numpy as np
import webdataset as wds
import zarr
from torch.utils.data import IterableDataset
from training.data import (
    ResampledShards2,
    expand_urls,
    filter_no_caption_or_no_image,
    log_and_continue,
)


def random_samples(sources, probs=None, longest=False, seed=None):
    """Yield samples randomly from multiple sources based on given probabilities.

    Args:
        sources (list): List of iterable sources to draw samples from.
        probs (list, optional): List of probabilities for each source. Defaults to None.
        longest (bool): If True, continue until all sources are exhausted. Defaults to False.

    Yields:
        Sample randomly selected from one of the sources.
    """
    if probs is None:
        probs = [1] * len(sources)
    else:
        probs = list(probs)
    while len(sources) > 0:
        cum = (np.array(probs) / np.sum(probs)).cumsum()
        r = np.random.default_rng(seed).random()
        seed += 1
        i = np.searchsorted(cum, r)
        print(i)
        try:
            yield next(sources[i])
        except StopIteration:
            if longest:
                del sources[i]
                del probs[i]
            else:
                break


class RandomMix(IterableDataset):
    """Iterate over multiple datasets by randomly selecting samples based on given probabilities."""

    def __init__(self, datasets, probs=None, longest=False, seed=None):
        """Initialize the RandomMix iterator.

        Args:
            datasets (list): List of datasets to iterate over.
            probs (list, optional): List of probabilities for each dataset. Defaults to None.
            longest (bool): If True, continue until all datasets are exhausted. Defaults to False.
        """
        self.datasets = datasets
        self.probs = probs
        self.longest = longest
        self.seed = seed

    def __iter__(self):
        """Return an iterator over the sources.

        Returns:
            iterator: An iterator that yields samples randomly from the datasets.
        """
        sources = [iter(d) for d in self.datasets]
        return random_samples(
            sources, self.probs, longest=self.longest, seed=self.seed
        )


def give_worker_shards(
    shards,
    worker_id,
    worker_total,
):
    # load the list of all UIDs
    metadata_files = [shard.replace(".tar", "_stats.json") for shard in shards]
    shard_file_counts = [
        json.load(open(metadata_file, "r"))["successes"]
        for metadata_file in metadata_files
    ]
    # Calculate cumulative counts to help with splitting
    cumsum = np.cumsum(shard_file_counts)
    total_samples = cumsum[-1]

    # Calculate target samples per worker
    target_per_worker = total_samples / worker_total
    print(target_per_worker)

    # Find which shards belong to this worker
    worker_start = target_per_worker * worker_id
    worker_end = target_per_worker * (worker_id + 1)

    # Find shard indices that contain the worker's samples
    start_shard = next(i for i, c in enumerate(cumsum) if c > worker_start)
    end_shard = next(i for i, c in enumerate(cumsum) if c >= worker_end)

    print(start_shard, end_shard)
    print(cumsum[end_shard] - cumsum[start_shard])
    return shards[start_shard : end_shard + 1], cumsum[end_shard] - cumsum[
        start_shard
    ]


def dataset_cfg_to_shards(dataset_cfg):
    shards = []
    if dataset_cfg.uris is None:
        dataset_cfg.uris = [dataset_cfg.uri]

    for uri in dataset_cfg.uris:
        if ".." in uri:
            uris, _ = expand_urls(uri)
            shards.extend(uris)
        else:
            shards.append(uri)
    return shards


def give_dataset(
    cfg,
    dataset_name,
    worker_id,
    worker_total,
    tokenizer,
    preprocess_val,
    batch_size,
    existing_uids=None,
):
    dataset_cfg = cfg.datasets[dataset_name]
    shards, num_samples = give_worker_shards(
        dataset_cfg_to_shards(dataset_cfg), worker_id, worker_total
    )
    dataset = wds.SimpleShardList(shards)
    pipeline = [dataset]
    pipeline.extend(
        [
            wds.split_by_worker,
            # tarfile_to_samples_nothrow,
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.decode("pilrgb", handler=log_and_continue),
        ]
    )
    if existing_uids is not None:

        def ood_filter(sample):
            uid = sample["json"]["uid"]
            return uid not in existing_uids

        pipeline.extend([wds.select(ood_filter)])

    pipeline.extend(
        [
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            wds.map_dict(
                image=preprocess_val,
                text=lambda text: tokenizer(text)[0],
                # json=lambda data: json.loads(data.decode("utf-8")),
            ),
            wds.to_tuple("image", "text", "json"),
            wds.batched(batch_size, partial=True),
        ]
    )
    dataset = wds.DataPipeline(*pipeline)
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=dataset_cfg.num_workers,
        persistent_workers=True,
    )
    num_batches = math.ceil(num_samples / batch_size)
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return dataloader


def give_embedding_dataset(
    cfg,
    dataset_name,
    id_dataset_name,
    tokenizer,
    preprocess_val,
    batch_size,
):
    ood_dataset_cfg = cfg.datasets[dataset_name]
    ood_dataset = ResampledShards2(
        ood_dataset_cfg.uri,
        deterministic=True,  # worker_seed=cfg.seed
    )  # same constrastive samples on all workers
    if id_dataset_name is not None:
        id_dataset_cfg = cfg.datasets[id_dataset_name]
        id_dataset = ResampledShards2(
            id_dataset_cfg.uri,
            deterministic=True,  # worker_seed=cfg.seed
        )
        if "fairvision/Glaucoma" in id_dataset_name:
            task = "fairvision/glaucoma"
        elif "fairvision/AMD" in id_dataset_name:
            task = "fairvision/amd"
        elif "fairvision/DR" in id_dataset_name:
            task = "fairvision/dr"
        elif "fitzpatrick17k" in id_dataset_name:
            task = "fitzpatrick17k"
        elif "pcam" in id_dataset_name:
            task = "pcam"
        else:
            raise ValueError(f"Unknown task: {id_dataset_name}")
        class_names = {
            "diabetic_retinopathy": [
                "no diabetic retinopathy",
                "mild diabetic retinopathy",
                "moderate diabetic retinopathy",
                "severe diabetic retinopathy",
                "proliferative diabetic retinopathy",
            ],
            "fairvision/amd": [
                "no age-related macular degeneration",
                "early age-related macular degeneration",
                "intermediate age-related macular degeneration",
                "late age-related macular degeneration",
            ],
            "fairvision/dr": [
                "no vision threatening diabetic retinopathy",
                "vision threatening diabetic retinopathy",
            ],
            "fairvision/glaucoma": ["without glaucoma", "with glaucoma"],
            "fitzpatrick17k": [
                "benign lesion",
                "malignant lesion",
                "non-neoplastic condition",
            ],
            "pcam": [
                "a histopathology slide showing lymph node",
                "histopathology image of lymph node containing metastatic tumor tissue",
            ],
        }
        templates = {
            "diabetic_retinopathy": ["a retinal image with {c}."],
            "fairvision/amd": ["a retinal image with {c}."],
            "fairvision/dr": ["a retinal image with {c}."],
            "fairvision/glaucoma": ["a retinal image {c}."],
            "fitzpatrick17k": ["a skin image showing a {c}."],
            "pcam": ["{c}."],
        }

        def label_map(label):
            return templates[task][0].format(c=class_names[task][int(label)])

        id_zarr = zarr.open("/raid/pdpl/id_downstream_idx.zarr", mode="r")
        id_uids = set(id_zarr[task]["id_indices"])

        def id_filter(sample):
            return sample["__key__"] in id_uids

        indistribution_data_num_samples = len(id_uids)

        p_id = indistribution_data_num_samples / (
            indistribution_data_num_samples + ood_dataset_cfg.num_samples
        )

        probs = [1 - p_id, p_id]

        pipelines = [
            [ood_dataset],
            [id_dataset],
        ]
    else:
        id_dataset = None
        pipelines = [[ood_dataset]]

    for i, pipeline in enumerate(pipelines):
        pipeline.extend(
            [
                wds.split_by_worker,
                wds.tarfile_to_samples(
                    handler=log_and_continue
                ),  # tarfile_to_samples_nothrow,
            ]
        )
        if i == 1:
            pipeline.extend([wds.select(id_filter)])
        pipeline.extend([wds.select(filter_no_caption_or_no_image)])
        pipeline.extend(
            [
                wds.shuffle(
                    bufsize=5000,
                    initial=1000,
                ),
            ]
        )

    if len(pipelines) > 1:
        pipeline = wds.RandomMix(
            [wds.DataPipeline(*pipeline) for pipeline in pipelines],
            probs=probs,
            longest=False,
        )
    else:
        pipeline = pipelines[0]
    pipeline.extend(
        [
            wds.decode("pilrgb", handler=log_and_continue),
            wds.map(
                lambda sample: {
                    **sample,
                    "txt": label_map(sample["label"])
                    if "txt" not in sample
                    else sample["txt"],
                }
            ),
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            wds.map_dict(
                image=preprocess_val, text=lambda text: tokenizer(text)[0]
            ),
            wds.to_tuple("image", "text"),
            wds.batched(batch_size, partial=True),
        ]
    )
    embedding_dataset = wds.DataPipeline(*pipeline)
    dataloader = wds.WebLoader(
        embedding_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=ood_dataset_cfg.num_workers,
        persistent_workers=True,
    )
    return dataloader
