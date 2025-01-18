import gc
import json
import logging
import os
import uuid
from typing import Dict, List, Optional

import hydra
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm

from compute_grads import Featurizer
from config.config import EncoderConfig, ExperimentConfig, register_configs
from data import give_dataset, give_embedding_dataset
from model import Model


class JsonFormatter(logging.Formatter):
    """Format log records as JSON."""

    def format(self, record):
        if isinstance(record.msg, dict):
            return json.dumps(record.msg)
        return json.dumps({"message": record.getMessage()})


def setup_logging():
    """Configure JSON logging to stdout."""
    logger = logging.getLogger("results")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    return logger


def log_results(params: Dict, metrics: Dict):
    """Log parameters and metrics as structured data."""
    logger = logging.getLogger("results")
    logger.info(
        {
            "type": "results",
            "params": params,
            "metrics": metrics,
        }
    )


def process_combination(
    cfg: DictConfig,
    experiment_cfg: ExperimentConfig,
    encoder_cfg: EncoderConfig,
    subworker_id: int,
    subworker_total: int,
):
    model = Model(encoder_cfg, cfg.device, cfg.s3_endpoint_url)
    model, tokenizer, _, preprocess_val = model.create_model_and_transforms()
    embeddings_dataset = give_embedding_dataset(
        cfg,
        experiment_cfg.ood_dataset_name,
        experiment_cfg.id_dataset_name,
        tokenizer,
        preprocess_val,
        encoder_cfg.embedding_batch_size,
    )
    featurizer = Featurizer(
        model=model,
        task="clip",
        proj_dim=cfg.proj_dim,
        device="cuda",
        model_id=encoder_cfg.model_id,
        use_half_precision=True,
        projector_seed=cfg.seed,
        proj_max_batch_size=encoder_cfg.grad_batch_size,
    )
    featurizer.task.get_embeddings(
        model,
        tqdm(
            embeddings_dataset,
            desc="Computing embeddings",
            total=cfg.num_contrastive_samples
            // encoder_cfg.embedding_batch_size
            + 1,
        ),
        batch_size=encoder_cfg.embedding_batch_size,
        size=cfg.num_contrastive_samples,
        embedding_dim=512,
        preprocess_fn_img=lambda x: x.to("cuda").to(torch.float16),
        preprocess_fn_txt=lambda x: x.to("cuda"),
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    for dataset_name in encoder_cfg.target_datasets:
        output_base_path = f"{cfg.output_dir}/{experiment_cfg.name}/{encoder_cfg.name}/{dataset_name.lower()}"
        os.makedirs(output_base_path, exist_ok=True)

        schema = pa.schema(
            [
                ("uid", pa.string()),
                ("grads", pa.list_(pa.float16(), cfg.proj_dim)),
                ("loss_grads", pa.float32()),
                ("loss", pa.float32()),
                ("loss_img", pa.float32()),
                ("loss_txt", pa.float32()),
            ]
        )

        # Check for existing UIDs across all partitions
        existing_uids = set()
        if os.path.exists(output_base_path):
            dataset = ds.dataset(output_base_path, format="parquet")
            for batch in dataset.scanner().to_batches():
                existing_uids.update(batch["uid"].to_pylist())

        print(f"Existing uids: {len(existing_uids)}")
        accumulated_tables = []
        write_interval = cfg.write_chunks

        metadata = {
            "grads_shape": str(cfg.proj_dim),
        }

        data = give_dataset(
            cfg,
            dataset_name,
            subworker_id,
            subworker_total,
            tokenizer,
            preprocess_val,
            encoder_cfg.grad_batch_size,
            existing_uids=existing_uids,
        )
        if data.num_samples == 0:
            print(f"No data for {dataset_name} and {subworker_id}")
            continue
        print(
            f"Data for {dataset_name} and {subworker_id} has {data.num_samples} samples (before filtering)"
        )

        partition_base_name = f"part_{subworker_id}"
        current_partition = 0

        for batch_idx, (img, txt, metadata_batch) in enumerate(
            tqdm(data, total=data.num_batches)
        ):
            uids = [m["uid"] for m in metadata_batch]

            img = img.to("cuda").to(torch.float16)
            txt = txt.to("cuda")

            grads, loss_grads, loss_img, loss_txt = featurizer.featurize(
                (img, txt)
            )
            batch_data = {
                "uid": uids,
                "grads": [
                    row.astype("float16") for row in grads.cpu().numpy()
                ],
                "loss_grads": loss_grads.cpu().numpy().astype("float32"),
                "loss_img": loss_img.cpu().numpy().astype("float32"),
                "loss_txt": loss_txt.cpu().numpy().astype("float32"),
                "loss": (loss_img + loss_txt).cpu().numpy().astype("float32"),
            }
            batch_table = pa.Table.from_pydict(batch_data, schema=schema)
            accumulated_tables.append(batch_table)

            # Write accumulated data at intervals
            if (batch_idx + 1) % write_interval == 0:
                if len(accumulated_tables) > 0:
                    combined_table = pa.concat_tables(accumulated_tables)
                    combined_table = combined_table.replace_schema_metadata(
                        metadata
                    )

                    output_path = f"{output_base_path}/{partition_base_name}_{current_partition}_final_{uuid.uuid4()}.parquet"
                    pq.write_table(combined_table, output_path)
                    print(
                        f"Wrote partition {current_partition} with {len(accumulated_tables)} batches"
                    )

                    current_partition += 1
                    accumulated_tables = []

        # Write any remaining data
        if len(accumulated_tables) > 0:
            combined_table = pa.concat_tables(accumulated_tables)
            combined_table = combined_table.replace_schema_metadata(metadata)

            output_path = f"{output_base_path}/{partition_base_name}_{current_partition}_{uuid.uuid4()}.parquet"
            pq.write_table(combined_table, output_path)
            print(
                f"Wrote final partition {current_partition} with {len(accumulated_tables)} batches"
            )


def get_worker_assignments(
    encoders: List[EncoderConfig],
    worker_id: int,
    worker_total: int,
) -> List[dict]:
    """Get all encoders that would be processed by this worker.

    Returns:
        List of dicts containing assignment details
    """
    # number of workers per encoder
    subworker_total = max(1, worker_total // len(encoders))

    assignments = []
    for i, encoder_cfg in enumerate(encoders):
        print(
            f"Encoder {i}: worker_id % len(encoders) = {worker_id % len(encoders)}"
        )
        print(f"Encoder {i}: i % worker_total = {i % worker_total}")
        if i % worker_total == worker_id % len(
            encoders
        ):  # should we work on this encoder?
            subworker_id = worker_id // len(
                encoders
            )  # what is our subworker id for this encoder?
            print(f"Encoder {i}: subworker_id = {subworker_id}")
            print(f"Encoder {i}: subworker_total = {subworker_total}")
            if subworker_id < subworker_total:
                assignments.append(
                    {
                        "encoder_id": i,
                        "encoder_name": encoder_cfg.name,
                        "subworker_id": subworker_id,
                        "subworker_total": subworker_total,
                    }
                )
    print(f"Subworker got {len(assignments)} assignments")
    return assignments


def run_worker(
    cfg: DictConfig,
    experiment: ExperimentConfig,
    dry_run: bool = False,
) -> Optional[List[dict]]:
    """Run or simulate worker processing based on configuration.

    Args:
        dry_run: If True, return assignments instead of processing them
    """
    assignments = get_worker_assignments(
        experiment.encoders, cfg.worker_id, cfg.worker_total
    )

    if dry_run:
        return assignments

    for assignment in assignments:
        if "dropped" in assignment:
            continue

        encoder_cfg = next(
            e
            for e in experiment.encoders
            if e.name == assignment["encoder_name"]
        )
        process_combination(
            cfg=cfg,
            experiment_cfg=experiment,
            encoder_cfg=encoder_cfg,
            subworker_id=assignment["subworker_id"],
            subworker_total=assignment["subworker_total"],
        )


@hydra.main(version_base=None, config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for the application."""
    setup_logging()

    # Check if this is a dry run
    dry_run = cfg.get("dry_run", False)

    for experiment in cfg.experiments:
        if dry_run:
            assignments = run_worker(cfg, experiment, dry_run=True)
            logger = logging.getLogger("results")
            logger.info(
                {
                    "type": "dry_run",
                    "worker_id": cfg.worker_id,
                    "worker_total": cfg.worker_total,
                    "assignments": assignments,
                }
            )
        else:
            run_worker(cfg, experiment)


if __name__ == "__main__":
    register_configs()
    main()
