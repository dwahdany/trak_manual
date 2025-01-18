from dataclasses import dataclass, field
from typing import List

from .config import EncoderConfig, ExperimentConfig


@dataclass
class Food101Experiment(ExperimentConfig):
    name: str = "food101"
    id_dataset_name: str = "Food101"
    ood_dataset_name: str = "commonpool"
    target_datasets: List[str] = field(
        default_factory=lambda: [
            "commonpool",
            "Food101",
        ]
    )
    encoders: List[EncoderConfig] = field(
        default_factory=lambda: [
            EncoderConfig(
                name="food101_v0",
                architecture="ViT-B-32",
                url="s3://pdpl/small_clip_checkpoints/curation/image-based/food101/ratio_1.0/datacomp_v0/small_scale/checkpoints/epoch_5.pt",
                model_id=0,
            ),
            EncoderConfig(
                name="food101_v1",
                architecture="ViT-B-32",
                url="s3://pdpl/small_clip_checkpoints/curation/image-based/food101/ratio_1.0/datacomp_v1/small_scale/checkpoints/epoch_5.pt",
                model_id=1,
            ),
            EncoderConfig(
                name="food101_v2",
                architecture="ViT-B-32",
                url="s3://pdpl/small_clip_checkpoints/curation/image-based/food101/ratio_1.0/datacomp_v2/small_scale/checkpoints/epoch_5.pt",
                model_id=2,
            ),
        ]
    )
