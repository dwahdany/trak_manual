from pathlib import Path

import numpy as np
import torch as ch
import zarr
from torch import Tensor
from tqdm.rich import tqdm


def load_ood_grads(cfg, encoder_cfg):
    input_path = str(
        Path(cfg.output_dir)
        / encoder_cfg.name
        / encoder_cfg.ood_dataset_name
        / "data.zarr"
    )
    dataset = zarr.open(input_path)
    uids = np.char.add(
        np.vectorize(lambda x: f"{x:016x}")(dataset.uid["f0"]),
        np.vectorize(lambda x: f"{x:016x}")(dataset.uid["f1"]),
    )
    g = zarr.load(input_path + "/grads")
    out_to_loss = zarr.load(input_path + "/loss_grads")
    # Create a structured array combining both arrays
    dtype = [
        ("uids", uids.dtype),
        ("grads", g.dtype, g.shape[1]),
        (
            "loss_grads",
            out_to_loss.dtype,
        ),
    ]
    combined = np.empty(len(uids), dtype=dtype)
    combined["uids"] = uids
    combined["grads"] = g
    combined["loss_grads"] = out_to_loss

    # Sort in-place based on uids
    combined.sort(order="uids")

    # Extract back the sorted arrays
    uids = ch.tensor(combined["uids"], device="cpu").pin_memory()
    g = ch.tensor(combined["grads"], device="cpu").pin_memory()
    out_to_loss = ch.tensor(combined["loss_grads"], device="cpu").pin_memory()
    return uids, g, out_to_loss


def get_xtx(grads: ch.Tensor, batch_size=20_000) -> Tensor:
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


def get_indices(target, id: bool = True):
    id_indices_zarr = zarr.open("/raid/pdpl/id_downstream_idx.zarr", mode="r")
    if id:
        return id_indices_zarr[target]["id_indices"]
    else:
        return id_indices_zarr[target]["downstream_indices"]


def compute_scores(features, out_to_loss, target):
    scores_zarr = zarr.open(
        "/datasets/datacomp/nearest_neighbor_scores.zarr", mode="a"
    )
    if "trak" not in scores_zarr:
        scores_zarr.create_group("trak")
    scores_zarr = scores_zarr["trak"]
    if target in scores_zarr:
        return
    input_path = str(Path(cfg.output_dir) / encoder_cfg.name / target)
    dataset_target = ds.dataset(input_path, format="parquet")
    batch_size = 16384
    scanner = dataset_target.scanner(
        columns=["grads", "uid"], batch_size=batch_size
    )
    batches = scanner.to_batches()
    grads_list = []
    uids_list = []
    for batch in tqdm(
        scanner.to_batches(), total=dataset_target.count_rows() // batch_size
    ):
        grads_list.extend(batch.column("grads").to_numpy(zero_copy_only=False))
        uids_list.extend(batch.column("uid").to_numpy(zero_copy_only=False))
    g_target = np.stack(grads_list)
    uids_target = np.stack(uids_list)
    dtype = [
        ("uids", uids_target.dtype),
        ("grads", g_target.dtype, g_target.shape[1]),
    ]
    combined = np.empty(len(uids_target), dtype=dtype)
    combined["uids"] = uids_target
    combined["grads"] = g_target
    combined.sort(order="uids")
    uids_target = combined["uids"]
    g_target = combined["grads"]
    id_indices = get_indices(target, id=False)  # get downstream indices
    g_target_pt = ch.tensor(g_target[id_indices], device="cpu").pin_memory()

    batch_size = 8192 * 2
    scores = []
    for i in trange(0, len(features), batch_size):
        batch = features_pt[i : i + batch_size].cuda()
        batch_scores = ch.mean(batch @ g_target_pt.cuda().T, axis=1)
        scores.append(batch_scores.cpu())
    scores = ch.cat(scores)
    scores = scores * out_to_loss
    if target not in scores_zarr:
        target_group = scores_zarr.create_group(target)
    else:
        target_group = scores_zarr[target]

    # Save the scores
    target_group.array(
        "id_scores", np.array(scores.cpu()), dtype=np.float32, overwrite=True
    )
    all_scores[target] = scores.cpu()


def main():
    out_to_loss = []
    uids = []
    features = []

    for encoder_cfg in cfg.encoders:
        uids_, g_, out_to_loss_ = load_ood_grads(cfg, encoder_cfg)
        uids.append(uids_)
        out_to_loss.append(out_to_loss_)
        xtx = get_xtx(g)
        x_xtx_inv = get_x_xtx_inv(g, xtx)
        features.append(x_xtx_inv)

    for i in range(1, len(uids)):
        assert (uids[0] == uids[i]).all(), (
            f"UIDs don't match between encoder 0 and encoder {i}"
        )
    uids = uids[0]

    out_to_loss = ch.stack(out_to_loss).mean(dim=0).pin_memory()
    features = ch.stack(features).pin_memory()
