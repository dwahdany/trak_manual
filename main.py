import gc
import json
import logging
import os
from typing import Dict, List, Optional

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from compute_grads import Featurizer
from config import EncoderConfig, register_configs
from data import give_dataset, give_embedding_dataset
from model import Model
from omegaconf import DictConfig
from tqdm.auto import tqdm


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
    encoder_cfg: EncoderConfig,
    subworker_id: int,
    subworker_total: int,
):
    model = Model(encoder_cfg, cfg.device)
    model, tokenizer, _, preprocess_val = model.create_model_and_transforms()
    embeddings_dataset = give_embedding_dataset(
        cfg,
        encoder_cfg.ood_dataset_name,
        encoder_cfg.id_dataset_name,
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

    output_path = f"{cfg.output_dir}/{encoder_cfg.name}/{encoder_cfg.ood_dataset_name}/grads_{subworker_id}.parquet"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = give_dataset(
        cfg,
        encoder_cfg.ood_dataset_name,
        subworker_id,
        subworker_total,
        tokenizer,
        preprocess_val,
        encoder_cfg.grad_batch_size,
    )

    schema = pa.schema(
        [
            ("uid", pa.string()),
            ("grads", pa.list_(pa.float16(), cfg.proj_dim)),
            ("loss_grads", pa.float32()),
        ]
    )

    metadata = {
        "grads_shape": str(cfg.proj_dim),
    }

    if not os.path.exists(output_path):
        table = pa.Table.from_pydict(
            {"uid": [], "grads": [], "loss_grads": []}, schema=schema
        ).replace_schema_metadata(metadata)
        pq.write_table(table, output_path)

    for img, txt, metadata in tqdm(data):
        uids = [m["uid"] for m in metadata]

        img = img.to("cuda").to(torch.float16)
        txt = txt.to("cuda")

        grads, loss_grads = featurizer.featurize((img, txt))

        batch_data = {
            "uid": uids,
            "grads": [row.astype("float16") for row in grads.cpu().numpy()],
            "loss_grads": loss_grads.cpu().numpy().astype("float32"),
        }
        batch_table = pa.Table.from_pydict(batch_data, schema=schema)

        if os.path.exists(output_path):
            with pq.ParquetWriter(output_path, schema=schema) as writer:
                writer.write_table(batch_table)
        else:
            pq.write_table(batch_table, output_path, compression="snappy")

        # del grads, loss_grads, batch_data, batch_table
        # torch.cuda.empty_cache()


def get_worker_assignments(
    encoders: List[EncoderConfig],
    worker_id: int,
    worker_total: int,
    dry_run: bool = False,
) -> List[dict]:
    """Get all encoders that would be processed by this worker.

    Returns:
        List of dicts containing assignment details
    """
    # number of workers per encoder
    subworker_total = worker_total // len(encoders)

    assignments = []
    for i, encoder_cfg in enumerate(encoders):
        if (
            i % worker_total == worker_id % subworker_total
        ):  # should we work on this encoder?
            subworker_id = worker_id // len(
                encoders
            )  # what is our subworker id for this encoder?
            if subworker_id < subworker_total:
                assignments.append(
                    {
                        "encoder_id": i,
                        "encoder_name": encoder_cfg.name,
                        "subworker_id": subworker_id,
                        "subworker_total": subworker_total,
                    }
                )
            # elif dry_run:
            #     assignments.append(
            #         {
            #             "encoder_name": encoder_name,
            #             "dataset_name": dataset_name,
            #             "dropped": True,
            #             "reason": f"Incomplete set: subworker {subworker_id} of {subworker_total}",
            #             "num_combinations": len(combinations),
            #         }
            #     )

    print(f"Subworker got {len(assignments)} assignments")
    return assignments


def run_worker(
    cfg: DictConfig,
    dry_run: bool = False,
) -> Optional[List[dict]]:
    """Run or simulate worker processing based on configuration.

    Args:
        dry_run: If True, return assignments instead of processing them
    """
    assignments = get_worker_assignments(
        cfg.encoders, cfg.worker_id, cfg.worker_total, dry_run
    )

    if dry_run:
        return assignments

    for assignment in assignments:
        if "dropped" in assignment:
            continue

        encoder_cfg = next(
            e for e in cfg.encoders if e.name == assignment["encoder_name"]
        )
        process_combination(
            cfg=cfg,
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

    if dry_run:
        assignments = run_worker(cfg, dry_run=True)
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
        run_worker(cfg)


if __name__ == "__main__":
    register_configs()
    main()
