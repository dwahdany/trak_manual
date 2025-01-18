from pathlib import Path
from pprint import pprint

import numpy as np
import torch as ch
import zarr
from torch import Tensor
from tqdm.rich import tqdm

from config.config import Config


def load_uids_grads(input_path):
    dataset = zarr.open(input_path)
    uids = np.char.add(
        np.vectorize(lambda x: f"{x:016x}")(dataset.uid["f0"]),
        np.vectorize(lambda x: f"{x:016x}")(dataset.uid["f1"]),
    )
    g = zarr.load(input_path + "/grads")

    # Create a structured array combining both arrays
    dtype = [("uids", uids.dtype), ("grads", g.dtype, g.shape[1])]
    combined = np.empty(len(uids), dtype=dtype)
    combined["uids"] = uids
    combined["grads"] = g

    # Sort in-place based on uids
    combined.sort(order="uids")

    # Extract back the sorted arrays
    uids = combined["uids"]
    g = combined["grads"]
    return uids, g


def get_xtx(grads: Tensor, batch_size=20_000) -> Tensor:
    proj_dim = grads.shape[1]
    result = ch.zeros(proj_dim, proj_dim, dtype=grads.dtype, device="cuda")
    blocks = ch.split(grads, split_size_or_sections=batch_size, dim=0)

    for block in tqdm(blocks):
        result += block.T.to("cuda") @ block.to("cuda")

    return result


def get_x_xtx_inv(
    grads: Tensor, xtx: Tensor, lambda_reg=0.0, batch_size=20_000
) -> Tensor:
    xtx_reg = xtx + lambda_reg * ch.eye(
        xtx.size(dim=0), device=xtx.device, dtype=xtx.dtype
    )
    xtx_inv = ch.linalg.inv(xtx_reg.to(ch.float32))

    # center X^TX inverse a bit to avoid numerical issues when going to float16
    xtx_inv /= xtx_inv.abs().mean()
    xtx_inv = xtx_inv.to(grads.dtype)

    grads_blocks = ch.split(grads, split_size_or_sections=batch_size, dim=0)

    # Move xtx_inv to GPU once before the loop
    xtx_inv_gpu = xtx_inv.cuda()

    # Process blocks on GPU
    result_blocks = []
    for block in tqdm(grads_blocks, desc="Processing blocks"):
        # Move block to GPU, compute, then back to CPU
        block_gpu = block.cuda()
        result_gpu = block_gpu @ xtx_inv_gpu
        result_blocks.append(result_gpu.cpu())

    # Concatenate results on CPU
    result = ch.cat(result_blocks)

    return result.to(dtype=grads.dtype)


def save_scores(scores, cfg, encoder_cfg, target_dataset_name, score_type):
    scores_path = str(Path(cfg.save_dir) / "scores.zarr")
    store = zarr.open(scores_path, mode="a")
    if encoder_cfg.id_dataset_name is None:
        id_suffix = "None"
    else:
        id_suffix = f"{encoder_cfg.id_dataset_name}"
    store.create_dataset(
        f"{encoder_cfg.name}/{encoder_cfg.ood_dataset_name}/{id_suffix}/{target_dataset_name}/{score_type}",
        data=scores,
        overwrite=True,
    )


def give_model_scores(encoder_cfg, cfg):
    input_path = str(
        Path(cfg.output_dir)
        / encoder_cfg.name
        / encoder_cfg.ood_dataset_name
        / "data.zarr"
    )
    uids, g = load_uids_grads(input_path)

    xtx = get_xtx(ch.tensor(g, device="cpu"))
    x_xtx_inv = get_x_xtx_inv(ch.tensor(g, device="cpu"), xtx)
    features = x_xtx_inv

    all_scores = {}
    for target_dataset_name in encoder_cfg.target_datasets:
        target_path = str(
            Path(cfg.output_dir)
            / encoder_cfg.name
            / target_dataset_name
            / "data.zarr"
        )
        target_uids, g_target = load_uids_grads(target_path)
        features_pinned = features.pin_memory()
        g_target_pinned = ch.tensor(g_target, device="cpu").pin_memory()

        from tqdm.rich import trange

        batch_size = 2048
        scores = []
        for i in trange(0, len(features), batch_size):
            batch = features_pinned[i : i + batch_size].cuda()
            batch_scores = ch.mean(batch @ g_target_pinned.cuda().T, axis=1)
            scores.append(batch_scores.cpu())
        scores = ch.cat(scores)

        all_scores[target_dataset_name] = scores
        save_scores(scores, cfg, encoder_cfg, target_dataset_name, "mean")

    return all_scores


def main():
    cfg = Config()
    # cfg.device="cpu"
    pprint(cfg)
    all_scores = []

    for encoder_cfg in cfg.encoders:
        scores = give_model_scores(encoder_cfg, cfg)
        all_scores.append(scores)

    for dataset_name in cfg.datasets.keys():
        dataset_scores = [
            scores[dataset_name]
            for scores in all_scores
            if dataset_name in scores
        ]
        if dataset_scores:  # Check if we have any scores for this dataset
            avg_scores = ch.stack(dataset_scores).mean(dim=0)
            print(
                f"{dataset_name}: {avg_scores.mean():.3f} ± {avg_scores.std():.3f}"
            )
