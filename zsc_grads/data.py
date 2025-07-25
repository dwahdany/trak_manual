import json
import logging
import math
from typing import Optional

import numpy as np
import webdataset as wds
from torch.utils.data import IterableDataset
from torchvision.datasets import CIFAR10, CIFAR100, STL10, Food101
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
        return random_samples(sources, self.probs, longest=self.longest, seed=self.seed)


def give_worker_shards(
    shards,
    worker_id,
    worker_total,
):
    metadata_files = [shard.replace(".tar", "_stats.json") for shard in shards]
    shard_file_counts = [
        json.load(open(metadata_file, "r"))["successes"]
        for metadata_file in metadata_files
    ]
    # shuffling the shards helps when changing the number of workers
    rng = np.random.default_rng(seed=worker_total)
    perm = rng.permutation(len(shards))
    shards = [shards[i] for i in perm]
    shard_file_counts = [shard_file_counts[i] for i in perm]
    cumsum = np.cumsum(shard_file_counts)
    total_samples = cumsum[-1]

    if worker_total == 1:
        if worker_id == 0:
            return shards, total_samples
        else:
            return [], 0
    if worker_id >= worker_total:
        return [], 0

    # Calculate target samples per worker
    target_per_worker = total_samples / worker_total
    print(f"Trying to give each worker {target_per_worker} samples")

    if worker_id == worker_total - 1:
        # Last worker gets all remaining shards
        start_shard = next(
            i for i, c in enumerate(cumsum) if c > worker_id * target_per_worker
        )
        samples = (
            total_samples - cumsum[start_shard - 1]
            if start_shard > 0
            else total_samples
        )
        print(f"Giving last worker {worker_id} shards {start_shard} to end")
        print(f"Giving last worker {worker_id} {samples} samples")
        return shards[start_shard:], samples
    else:
        # Other workers get shards up until next worker's start
        worker_start = target_per_worker * worker_id
        next_worker_start = target_per_worker * (worker_id + 1)

        start_shard = next(i for i, c in enumerate(cumsum) if c > worker_start)
        end_shard = next(i for i, c in enumerate(cumsum) if c > next_worker_start)

        samples = cumsum[end_shard - 1] - (
            cumsum[start_shard - 1] if start_shard > 0 else 0
        )
        print(f"Giving worker {worker_id} shards {start_shard} to {end_shard}")
        print(f"Giving worker {worker_id} {samples} samples")

        return shards[start_shard:end_shard], samples


def give_dataset_size(dataset_cfg):
    shards = dataset_cfg_to_shards(dataset_cfg)
    return sum(
        json.load(open(shard.replace(".tar", "_stats.json"), "r"))["successes"]
        for shard in shards
    )


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
    map_labels=True,
    include_json=True,
    dataset_kwargs: Optional[dict] = None,
):
    dataset_cfg = cfg.datasets[dataset_name]
    if not dataset_cfg.splittable:
        if worker_id > 0:
            return None
        shards, num_samples = give_worker_shards(
            dataset_cfg_to_shards(dataset_cfg), 0, 1
        )  # use just one worker
    else:
        shards, num_samples = give_worker_shards(
            dataset_cfg_to_shards(dataset_cfg), worker_id, worker_total
        )
    if dataset_cfg.custom:
        dataset, label_map = give_custom_dataset(
            cfg, dataset_name, uri=shards, resampled=False, dataset_kwargs=dataset_kwargs
        )
    else:
        dataset = wds.SimpleShardList(shards)
    pipeline = [dataset]
    pipeline.extend(
        [
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.decode("pilrgb", handler=log_and_continue),
        ]
    )
    if dataset_cfg.custom:
        pipeline.extend(
            [
                wds.map(
                    lambda sample: {
                        **sample,
                        "json": {
                            **sample["json"],
                            "uid": str(sample["json"].get("id")),
                        }
                        if "id" in sample["json"]
                        else sample["json"],
                    }
                ),
            ]
        )
    if existing_uids is not None:

        def ood_filter(sample):
            try:
                uid = sample["json"]["uid"]
                return uid not in existing_uids
            except KeyError as e:
                print(f"Didnt find uid in {sample}")
                raise e

        pipeline.extend([wds.select(ood_filter)])

    if map_labels:
        pipeline.extend(
            [   
                wds.map(
                    lambda sample: {
                        **sample,
                        "txt": label_map(sample["label"])
                        if "txt" not in sample
                        else sample["txt"],
                    }
                ),
            ])
    else:
        pipeline.extend([
            wds.map(
                lambda sample: {
                    **sample,
                    "txt": int(sample["label"])
                    if "txt" not in sample
                    else sample["txt"],
                }
            ),
        ])
    pipeline.extend([wds.rename(image="jpg;png;jpeg;webp", text="txt")])
    pipeline.extend([
        wds.map_dict(
            image=preprocess_val,
        ),
    ])
    if tokenizer is not None:
        pipeline.extend([
            wds.map_dict(
                text=lambda text: tokenizer(text)[0],
                # json=lambda data: json.loads(data.decode("utf-8")),
            ),
        ])
    else:
        logging.warning("No tokenizer provided, using raw text")
    if include_json:
        pipeline.extend([
                wds.to_tuple("image", "text", "json"),
                wds.batched(batch_size, partial=True),
            ]
        )
    else:
        pipeline.extend([
            wds.to_tuple("image", "text"),
            wds.batched(batch_size, partial=True),
        ])
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


def give_zeroshot_sets(
    cfg,
    id_dataset_name: str,
    dataset_kwargs: Optional[dict] = None,
):
    if "fairvision/glaucoma" in id_dataset_name.lower():
        task = "fairvision/glaucoma"
    elif "fairvision/amd" in id_dataset_name.lower():
        task = "fairvision/amd"
    elif "fairvision/dr" in id_dataset_name.lower():
        task = "fairvision/dr"
    elif "fitzpatrick17k" in id_dataset_name.lower():
        task = "fitzpatrick17k"
    elif "pcam" in id_dataset_name.lower():
        task = "pcam"
    elif "food101" in id_dataset_name.lower():
        task = "food101"
    elif "cifar100" in id_dataset_name.lower():
        task = "cifar100"
    elif "cifar10" in id_dataset_name.lower():
        task = "cifar10"
    elif "stl10" in id_dataset_name.lower():
        task = "stl10"
    elif "resisc45" in id_dataset_name.lower():
        task = "resisc45"
    else:
        raise ValueError(f"Unknown task: {id_dataset_name}")
    ds_class = {
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
        "food101": Food101,
        "stl10": STL10,
    }
    if task.lower() in ds_class.keys():
        if dataset_kwargs is None:
            dataset_kwargs = {}
        classnames = ds_class[task.lower()](root=cfg.datasets[id_dataset_name].root, download=True, **dataset_kwargs).classes
    else:
        with open(
            "/home/c02dawa/CISPA-az6/pdpl-2025/code/CLIP_benchmark/clip_benchmark/datasets/en_classnames.json"
        ) as f:
            classnames = json.load(f)[task.lower()]

    with open(
        "/home/c02dawa/CISPA-az6/pdpl-2025/code/CLIP_benchmark/clip_benchmark/datasets/en_zeroshot_classification_templates.json"
    ) as f:
        templates = json.load(f)[task.lower()]

    return classnames, templates

def give_custom_dataset(
    cfg,
    id_dataset_name: str,
    uri: Optional[str] = None,
    resampled=True,
    dataset_kwargs: Optional[dict] = None,
):
    shards = cfg.datasets[id_dataset_name].uri
    if uri is not None:
        shards = uri
    if resampled:
        id_dataset = ResampledShards2(
            shards,
            deterministic=True,  # worker_seed=cfg.seed
        )
    else:
        id_dataset = wds.SimpleShardList(shards)
    if "fairvision/glaucoma" in id_dataset_name.lower():
        task = "fairvision/glaucoma"
    elif "fairvision/amd" in id_dataset_name.lower():
        task = "fairvision/amd"
    elif "fairvision/dr" in id_dataset_name.lower():
        task = "fairvision/dr"
    elif "fitzpatrick17k" in id_dataset_name.lower():
        task = "fitzpatrick17k"
    elif "pcam" in id_dataset_name.lower():
        task = "pcam"
    elif "food101" in id_dataset_name.lower():
        task = "food101"
    elif "cifar100" in id_dataset_name.lower():
        task = "cifar100"
    elif "cifar10" in id_dataset_name.lower():
        task = "cifar10"
    elif "stl10" in id_dataset_name.lower():
        task = "stl10"
    elif "resisc45" in id_dataset_name.lower():
        task = "resisc45"
    else:
        raise ValueError(f"Unknown task: {id_dataset_name}")
    ds_class = {
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
        "food101": Food101,
        "stl10": STL10,
    }
    if task.lower() in ds_class.keys():
        if dataset_kwargs is None:
            dataset_kwargs = {}
        classnames = ds_class[task.lower()](root=cfg.datasets[id_dataset_name].root, download=True, **dataset_kwargs).classes
    else:
        with open(
            "/home/c02dawa/CISPA-az6/pdpl-2025/code/CLIP_benchmark/clip_benchmark/datasets/en_classnames.json"
        ) as f:
            classnames = json.load(f)[task.lower()]

    with open(
        "/home/c02dawa/CISPA-az6/pdpl-2025/code/CLIP_benchmark/clip_benchmark/datasets/en_zeroshot_classification_templates.json"
    ) as f:
        templates = json.load(f)[task.lower()]

    filled_templates = [
        [template.replace("{c}", c) for c in classnames] for template in templates
    ]

    def label_map(label, one=True):
        if one:
            return filled_templates[0][int(label)]
        else:
            return filled_templates[int(label)]

    return id_dataset, label_map


def give_number_of_samples(dataset_cfg):
    shards = dataset_cfg_to_shards(dataset_cfg)
    total_samples = 0
    for shard in shards:
        stats_file = shard.replace(".tar", "_stats.json")
        try:
            with open(stats_file) as f:
                stats = json.load(f)
                total_samples += stats["successes"]
        except FileNotFoundError:
            print(f"Warning: Stats file not found: {stats_file}")
            continue
    return total_samples


def give_embedding_dataset(
    cfg,
    dataset_name,
    tokenizer,
    preprocess_val,
    batch_size,
):
    ood_dataset_cfg = cfg.datasets[dataset_name]
    ood_dataset = ResampledShards2(
        ood_dataset_cfg.uri,
        deterministic=True,  # worker_seed=cfg.seed
    )  # same constrastive samples on all workers

    pipeline = [ood_dataset]

    pipeline.extend(
        [
            wds.split_by_worker,
            wds.tarfile_to_samples(
                handler=log_and_continue
            ),  # tarfile_to_samples_nothrow,
        ]
    )
    pipeline.extend([wds.select(filter_no_caption_or_no_image)])
    pipeline.extend(
        [
            wds.shuffle(
                bufsize=5000,
                initial=1000,
            ),
        ]
    )

    pipeline.extend(
        [
            wds.decode("pilrgb", handler=log_and_continue),
            wds.rename(image="jpg;png;jpeg;webp", text="txt"),
            wds.map_dict(image=preprocess_val, text=lambda text: tokenizer(text)[0]),
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
