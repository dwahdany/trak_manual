from dataclasses import dataclass, field
from typing import List

from config.config import EncoderConfig, ExperimentConfig


@dataclass
class DRExperiment(ExperimentConfig):
    name: str = "fairvision/dr"
    id_dataset_name: str = "fairvision/DR"
    ood_dataset_name: str = "commonpool"
    target_datasets: List[str] = field(
        default_factory=lambda: [
            "commonpool",
            "fairvision/DR",
        ]
    )
    encoders: List[EncoderConfig] = field(
        default_factory=lambda: [
            EncoderConfig(
                name="dr_v0",
                architecture="ViT-B-32",
                url="s3://pdpl/small_clip_checkpoints/curation/image-based/fairvision/dr/ratio_1.0/datacomp_v0/small_scale/checkpoints/epoch_5.pt",
                model_id=0,
            ),
            EncoderConfig(
                name="dr_v1",
                architecture="ViT-B-32",
                url="s3://pdpl/small_clip_checkpoints/curation/image-based/fairvision/dr/ratio_1.0/datacomp_v1/small_scale/checkpoints/epoch_5.pt",
                model_id=1,
            ),
            EncoderConfig(
                name="dr_v2",
                architecture="ViT-B-32",
                url="s3://pdpl/small_clip_checkpoints/curation/image-based/fairvision/dr/ratio_1.0/datacomp_v2/small_scale/checkpoints/epoch_5.pt",
                model_id=2,
            ),
        ]
    )
