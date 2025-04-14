from dataclasses import dataclass, field
from typing import Dict, List, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class DatasetConfig:
    uri: Optional[str] = None
    uris: Optional[List[str]] = None
    size: Optional[int] = None
    num_workers: int = 16
    splittable: bool = True
    custom: bool = False
    num_samples: Optional[int] = None

    def __post_init__(self):
        if self.uri is not None and self.uris is not None:
            raise ValueError("Only one of uri or uris should be set")
        if self.uri is None and self.uris is None:
            raise ValueError("Either uri or uris must be set")


@dataclass
class EncoderConfig:
    architecture: str
    name: str
    path: Optional[str] = None
    url: Optional[str] = None
    precision: str = "pure_fp16"  # "amp"
    embedding_batch_size: int = 4_096
    grad_batch_size: int = 120
    model_id: int = 0

    def __post_init__(self):
        if not (self.path or self.url):
            raise ValueError("Either path or url must be supplied")


@dataclass
class ExperimentConfig:
    name: str
    ood_dataset_name: str = "commonpool"
    id_dataset_name: Optional[str] = None
    target_datasets: List[str] = field(
        default_factory=lambda: [
            "commonpool",
            "fairvision/amd",
            # "fairvision/Glaucoma",
            # "fairvision/DR",
            # "fitzpatrick17k",
            "food101",
            "pcam",
            "cifar100",
            # "STL10",
        ],
    )
    encoders: List[EncoderConfig] = field(
        default_factory=lambda: [
            EncoderConfig(
                name="local_commonpool_s_s13m_b4k_0",
                architecture="ViT-B-32",
                path="/raid/pdpl/small_clip_checkpoints/raw/datacomp_v0/small_scale/checkpoints/epoch_5.pt",
                model_id=0,
            ),
            EncoderConfig(
                name="local_commonpool_s_s13m_b4k_1",
                architecture="ViT-B-32",
                path="/raid/pdpl/small_clip_checkpoints/raw/datacomp_v1/small_scale/checkpoints/epoch_5.pt",
                model_id=1,
            ),
            EncoderConfig(
                name="local_commonpool_s_s13m_b4k_2",
                architecture="ViT-B-32",
                path="/raid/pdpl/small_clip_checkpoints/raw/datacomp_v2/small_scale/checkpoints/epoch_5.pt",
                model_id=2,
            ),
        ]
    )


@dataclass
class Config:
    device: str = "cuda"
    worker_id: int = 0
    worker_total: int = 1
    dry_run: bool = False
    debug: bool = False
    output_dir: str = "/raid/pdpl/trak/grads/"
    save_dir: str = "/raid/pdpl/trak/trak_results/"
    s3_endpoint_url: Optional[str] = "https://s3.fraunhofer.de"
    write_chunks: int = 1000  # number of samples per output chunk
    seed: int = 42  # for sampling the constrastive samples, and projector
    proj_dim: int = 2048
    num_contrastive_samples: int = 50_000
    datasets: Dict[str, DatasetConfig] = field(
        default_factory=lambda: {
            "commonpool": DatasetConfig(
                uri="/datasets/datacomp/shards/{00000000..00001287}.tar",
                num_samples=10367394,
            ),
            "pcam": DatasetConfig(
                uri="/datasets/pcam/shards/pcam-train-{000000..000262}.tar",
                custom=True,
                num_samples=262144,
            ),
            "fairvision/amd": DatasetConfig(
                uri="/datasets/fairvision/AMD/shards/amd-train-{000000..000005}.tar",
                custom=True,
                num_samples=6000,
            ),
            "fairvision/glaucoma": DatasetConfig(
                uri="/datasets/fairvision/Glaucoma/shards/glaucoma-train-{000000..000005}.tar",
                custom=True,
                num_samples=6000,
            ),
            "fairvision/dr": DatasetConfig(
                uri="/datasets/fairvision/DR/shards/dr-train-{000000..000005}.tar",
                custom=True,
                num_samples=6000,
            ),
            "fitzpatrick17k": DatasetConfig(
                uri="/datasets/fitzpatrick17k/shards/fitzpatrick17k-train-{000000..000012}.tar",
                custom=True,
                num_samples=12858,
            ),
            "food101": DatasetConfig(
                uri="/datasets/food101/shards/food101-train-{000000..000075}.tar",
                custom=True,
                num_samples=75750,
            ),
            "cifar100": DatasetConfig(
                uri="/datasets/cifar100/shards/cifar100-train-{000000..000049}.tar",
                custom=True,
                num_samples=50000,
            ),
            "stl10": DatasetConfig(
                uri="/datasets/stl10/shards/stl10-train-{000000..000004}.tar",
                custom=True,
            ),
        }
    )
    experiments: List[ExperimentConfig] = field(
        default_factory=lambda: [
            ExperimentConfig(
                name="raw",
            ),
            # create_downstream_experiment("cifar100"),
            # create_downstream_experiment("stl10"),
            # create_downstream_experiment("food101"),
            # create_downstream_experiment("fitzpatrick17k"),
            # create_downstream_experiment("pcam"),
            # create_downstream_experiment("fairvision/amd"),
            # create_downstream_experiment("fairvision/glaucoma"),
            # create_downstream_experiment("fairvision/dr"),
        ]
    )


def create_downstream_experiment(name: str):
    return ExperimentConfig(
        name=name,
        id_dataset_name=name,
        ood_dataset_name="commonpool",
        target_datasets=["commonpool", name],
        encoders=[
            EncoderConfig(
                name=f"{name}_v0",
                architecture="ViT-B-32",
                path=f"s3://pdpl/small_clip_checkpoints/curation/random/{name.replace('amd', 'AMD').replace('glaucoma', 'Glaucoma').replace('dr', 'DR')}/ratio_1.0/datacomp_v0/small_scale/checkpoints/epoch_5.pt",
                model_id=0,
            ),
            EncoderConfig(
                name=f"{name}_v1",
                architecture="ViT-B-32",
                path=f"s3://pdpl/small_clip_checkpoints/curation/random/{name.replace('amd', 'AMD').replace('glaucoma', 'Glaucoma').replace('dr', 'DR')}/ratio_1.0/datacomp_v1/small_scale/checkpoints/epoch_5.pt",
                model_id=1,
            ),
            EncoderConfig(
                name=f"{name}_v2",
                architecture="ViT-B-32",
                path=f"s3://pdpl/small_clip_checkpoints/curation/random/{name.replace('amd', 'AMD').replace('glaucoma', 'Glaucoma').replace('dr', 'DR')}/ratio_1.0/datacomp_v2/small_scale/checkpoints/epoch_5.pt",
                model_id=2,
            ),
        ],
    )


def create_raw_experiments(seeds: List[int]):
    return ExperimentConfig(
        name="raw",
        encoders=[
            EncoderConfig(
                name=f"local_commonpool_s_s13m_b4k_{seed}",
                architecture="ViT-B-32",
                path=f"/raid/pdpl/small_clip_checkpoints/raw/datacomp_v{seed}/small_scale/checkpoints/epoch_5.pt",
                model_id=seed,
            )
            for seed in seeds
        ],
    )


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    cs.store(name="encoder", node=EncoderConfig)
    cs.store(name="dataset", node=DatasetConfig)
