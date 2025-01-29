import gc
import os
from pathlib import Path

import hydra
import torch
from compute_scores import PreComputedGradients
from config.config import EncoderConfig, register_configs
from model import Model
from omegaconf import DictConfig
import pyarrow.dataset as ds
from tqdm.rich import tqdm
import pyarrow.fs as fs

s3 = fs.S3FileSystem(
        endpoint_override="https://s3.fraunhofer.de"
    )

def process_encoders(cfg: DictConfig):
    """Process a single encoder configuration for multiple model IDs."""
    # Initialize model (needed for checkpoint loading)
    encoder_cfg = cfg.encoders[0]
    model = Model(cfg.encoders[0], cfg.device)
    model, _, _, _ = model.create_model_and_transforms()

    # Load pre-computed gradients from parquet files
    input_path = str(Path(cfg.output_dir) / encoder_cfg.name / encoder_cfg.ood_dataset_name).replace("s3:/","s3://")
    print(f"Input path: {input_path}")
    dataset = ds.dataset(input_path, format="parquet", filesystem=s3)
    table = dataset.to_table()
    train_set_size = len(table["uid"])
    print(f"Train set size: {train_set_size}")
    # Initialize PreComputedGradients
    pre_computed = PreComputedGradients(
        model=model,
        task="clip",
        train_set_size=train_set_size,
        device=cfg.device,
        proj_dim=cfg.proj_dim,
        use_half_precision=True,
    )
    
    # Process each model ID
    for encoder_cfg in tqdm(cfg.encoders, desc="Processing encoders"):
        input_path = f"{cfg.output_dir}/{encoder_cfg.name}/{encoder_cfg.ood_dataset_name}"
        pre_computed.load_parquet_data(input_path)
        
        model = Model(encoder_cfg, cfg.device)
        model, _, _, _ = model.create_model_and_transforms()
        print(f"Processing model ID: {encoder_cfg.model_id}")
        
        # Load checkpoint for this model ID
        checkpoint = torch.load(encoder_cfg.path)
        pre_computed.load_checkpoint(checkpoint, encoder_cfg.model_id)
        
        # Process all samples at once
        pre_computed.featurize(
            batch_indices=range(train_set_size),
            num_samples=train_set_size
        )
        
        # Finalize features for this model ID
        pre_computed.finalize_features(model_ids=[encoder_cfg.model_id])

        for dataset in cfg.datasets:
            traker.start_scoring_checkpoint(
                exp_name=f"{encoder_cfg.name}_for_{dataset}",
                checkpoint=checkpoint,
                model_id=encoder_cfg.model_id,
                num_targets=train_set_size
            )
    
    # Clean up
    del model, pre_computed
    gc.collect()
    torch.cuda.empty_cache()


@hydra.main(version_base=None, config_name="config")
def main(cfg: DictConfig) -> None:
    process_encoders(cfg)


if __name__ == "__main__":
    register_configs()
    main()
