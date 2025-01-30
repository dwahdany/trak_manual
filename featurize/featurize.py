from pathlib import Path
from pprint import pprint

import numpy as np
import pyarrow.dataset as ds
import torch as ch
import zarr
from config.config import Config, ExperimentConfig
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from torch import Tensor

DEBUG = False


def load_ood_grads(cfg, experiment_cfg, encoder_cfg, progress=None):
    if progress is not None:
        load_task = progress.add_task(
            f"Loading OOD grads for {encoder_cfg.name}...", total=3
        )

    input_path = str(
        Path(cfg.output_dir)
        / experiment_cfg.name
        / encoder_cfg.name
        / experiment_cfg.ood_dataset_name
        / "data.zarr"
    )
    dataset = zarr.open(input_path)

    zarr_chunk_size = dataset["uid"].chunks[0]
    chunk_size = zarr_chunk_size * 10
    total_size = dataset["uid"].shape[0] if not DEBUG else 1000
    chunks = []
    if progress is not None:
        chunk_task = progress.add_task(
            "Loading chunks...",
            total=(total_size + chunk_size - 1) // chunk_size,
        )

    for start_idx in range(0, total_size, chunk_size):
        end_idx = min(start_idx + chunk_size, total_size)

        # Load chunk data
        uids_chunk = dataset["uid"][start_idx:end_idx].astype("U32")
        g_chunk = dataset["grads"][start_idx:end_idx]
        out_to_loss_chunk = dataset["loss_grads"][start_idx:end_idx]

        # Create structured array for this chunk
        dtype = [
            ("uids", uids_chunk.dtype),
            ("grads", g_chunk.dtype, g_chunk.shape[1]),
            ("loss_grads", out_to_loss_chunk.dtype),
        ]
        chunk = np.empty(len(uids_chunk), dtype=dtype)
        chunk["uids"] = uids_chunk
        chunk["grads"] = g_chunk
        chunk["loss_grads"] = out_to_loss_chunk
        chunks.append(chunk)

        if progress is not None:
            progress.advance(chunk_task)

    if progress is not None:
        progress.remove_task(chunk_task)

    if progress is not None:
        progress.advance(load_task)

    # Merge and sort chunks
    combined = np.concatenate(chunks)
    combined.sort(order="uids")

    if progress is not None:
        progress.advance(load_task)

    # Extract arrays using buffer protocol to avoid extra copies
    uids = combined["uids"]
    g = ch.from_numpy(combined["grads"]).pin_memory()
    out_to_loss = ch.from_numpy(combined["loss_grads"]).pin_memory()

    if progress is not None:
        progress.advance(load_task)
        progress.remove_task(load_task)

    return uids, g, out_to_loss


def load_dataset_size(cfg, experiment_cfg, encoder_cfg):
    if DEBUG:
        return 1000
    input_path = str(
        Path(cfg.output_dir)
        / experiment_cfg.name
        / encoder_cfg.name
        / experiment_cfg.ood_dataset_name
        / "data.zarr"
    )
    try:
        dataset = zarr.open(input_path)
        ood_grads = dataset["grads"].shape[0]
        if experiment_cfg.id_dataset_name is not None:
            id_grads = get_indices(
                experiment_cfg.id_dataset_name, id=False
            ).shape[0]
            return ood_grads + id_grads
        else:
            return ood_grads
    except Exception as e:
        print(
            f"Error loading dataset size from {input_path} for {experiment_cfg.name} {encoder_cfg.name} {experiment_cfg.ood_dataset_name}"
        )
        raise e


def get_xtx(grads: Tensor, batch_size=20_000, progress=None) -> Tensor:
    proj_dim = grads.shape[1]
    result = ch.zeros(proj_dim, proj_dim, dtype=grads.dtype, device="cuda")
    blocks = ch.split(grads, split_size_or_sections=batch_size, dim=0)

    # Use progress.track if progress bar is provided, otherwise use regular iteration
    iterator = (
        progress.track(blocks, description="Computing XTX")
        if progress
        else blocks
    )
    for block in iterator:
        result += block.T.to("cuda") @ block.to("cuda")

    return result


def get_x_xtx_inv(
    grads: Tensor,
    xtx: Tensor,
    lambda_reg=0.0,
    batch_size=20_000,
    progress=None,
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
    # Use progress.track if progress bar is provided
    iterator = (
        progress.track(grads_blocks, description="Processing blocks")
        if progress
        else grads_blocks
    )
    for block in iterator:
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


def featurize_with_id(cfg, experiment_cfg):
    train_dataset_size = load_dataset_size(
        cfg, experiment_cfg, experiment_cfg.encoders[0]
    )
    id_indices = get_indices(experiment_cfg.id_dataset_name, id=True)

    avg_out_to_loss = ch.zeros(train_dataset_size, device="cpu")
    avg_scores = ch.zeros(train_dataset_size, device="cpu")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[{task.completed}/{task.total}]"),
        TimeElapsedColumn(),
    ) as progress:
        encoder_task = progress.add_task(
            "Processing encoders...", total=len(experiment_cfg.encoders)
        )

        for encoder_cfg in experiment_cfg.encoders:
            uids, g, out_to_loss_ood = load_ood_grads(
                cfg, experiment_cfg, encoder_cfg, progress=progress
            )
            assert len(g) == train_dataset_size - len(id_indices)
            assert len(out_to_loss_ood) == train_dataset_size - len(id_indices)

            input_path = str(
                Path(cfg.output_dir)
                / experiment_cfg.name
                / encoder_cfg.name
                / experiment_cfg.id_dataset_name
            )
            dataset_target = ds.dataset(input_path, format="parquet")
            batch_size = 16384
            scanner = dataset_target.scanner(
                columns=["grads", "uid"], batch_size=batch_size
            )
            grads_list = []
            uids_list = []

            out_to_loss_list = []

            batch_task = progress.add_task(
                f"Loading batches for {experiment_cfg.id_dataset_name}...",
                total=dataset_target.count_rows() // batch_size,
            )

            for batch in scanner.to_batches():
                grads_list.extend(
                    batch.column("grads").to_numpy(zero_copy_only=False)
                )
                uids_list.extend(
                    batch.column("uid").to_numpy(zero_copy_only=False)
                )
                out_to_loss_list.extend(
                    batch.column("loss_grads").to_numpy(zero_copy_only=False)
                )
                progress.advance(batch_task)

            progress.remove_task(batch_task)

            g_target = np.stack(grads_list)
            uids_target = np.stack(uids_list)
            out_to_loss_target = np.stack(out_to_loss_list)
            dtype = [
                ("uids", uids_target.dtype),
                ("grads", g_target.dtype, g_target.shape[1]),
                ("out_to_loss", out_to_loss_target.dtype),
            ]
            combined = np.empty(len(uids_target), dtype=dtype)
            combined["uids"] = uids_target
            combined["grads"] = g_target
            combined["out_to_loss"] = out_to_loss_target
            combined.sort(order="uids")
            uids_target = combined["uids"]
            g_target = combined["grads"]
            out_to_loss_target = combined["out_to_loss"]

            mask = np.isin(uids_target, id_indices)

            out_to_loss_id = out_to_loss_target[mask]
            g_train_pt = ch.cat([g, g_target[mask]])
            out_to_loss_train = np.concatenate(
                [out_to_loss_ood, out_to_loss_id]
            )
            avg_out_to_loss += out_to_loss_train

            xtx = get_xtx(
                ch.tensor(g_train_pt, device="cpu"), progress=progress
            )
            x_xtx_inv = get_x_xtx_inv(
                ch.tensor(g_train_pt, device="cpu"), xtx, progress=progress
            )
            features = x_xtx_inv.pin_memory()
            g_downstream_pt = ch.tensor(
                g_target[~mask], device="cpu"
            ).pin_memory()

            batch_size = 8192 * 2
            scores_downstream = []

            score_task = progress.add_task(
                f"Computing scores for {experiment_cfg.id_dataset_name}...",
                total=len(features) // batch_size + 1,
            )

            for i in range(0, len(features), batch_size):
                batch = features[i : i + batch_size].cuda()
                batch_scores = ch.mean(
                    batch @ g_downstream_pt.cuda().T, axis=1
                )
                scores_downstream.append(batch_scores.cpu())
                progress.advance(score_task)

            progress.remove_task(score_task)
            scores_downstream = ch.cat(scores_downstream)
            avg_scores += scores_downstream

            progress.advance(encoder_task)

    avg_out_to_loss /= len(experiment_cfg.encoders)
    avg_scores /= len(experiment_cfg.encoders)
    final_scores = (avg_scores * avg_out_to_loss).numpy()
    ood_scores = final_scores[: train_dataset_size - len(id_indices)]
    id_scores = final_scores[train_dataset_size - len(id_indices) :]

    # Create the zarr store
    store = zarr.open("/raid/pdpl/trak_scores.zarr", mode="a")

    # Get or create groups
    exp_group = store.require_group(experiment_cfg.name)
    encoder_group = exp_group.require_group(encoder_cfg.name)

    target_group = encoder_group.require_group(experiment_cfg.id_dataset_name)
    target_group.create_dataset(
        name="ood_scores",
        data=ood_scores,
        dtype=ood_scores.dtype,
        overwrite=True,
    )
    target_group.create_dataset(
        name="id_scores",
        data=id_scores,
        dtype=id_scores.dtype,
        overwrite=True,
    )


def featurize_no_id(
    cfg,
    experiment_cfg,
):
    print("Featurizing no ID")
    train_dataset_size = load_dataset_size(
        cfg, experiment_cfg, experiment_cfg.encoders[0]
    )
    targets = [
        "pcam",
        # "fitzpatrick17k",
        # "fairvision/dr",
        "fairvision/amd",
        # "fairvision/glaucoma",
        "food101",
        "cifar100",
        # "stl10",
    ]
    avg_out_to_loss = ch.zeros(train_dataset_size, device="cpu")
    avg_scores = {
        k: ch.zeros(train_dataset_size, device="cpu") for k in targets
    }
    uids = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[{task.completed}/{task.total}]"),
        TimeElapsedColumn(),
    ) as progress:
        encoder_task = progress.add_task(
            "Processing encoders...", total=len(cfg.experiments[0].encoders)
        )

        for encoder_cfg in cfg.experiments[0].encoders:
            # all_scores = {}
            uids, g, out_to_loss = load_ood_grads(
                cfg, cfg.experiments[0], encoder_cfg, progress=progress
            )
            if uids is None:
                uids = uids
            else:
                assert np.array_equal(uids, uids)
            avg_out_to_loss += out_to_loss
            xtx = get_xtx(ch.tensor(g, device="cpu"), progress=progress)
            x_xtx_inv = get_x_xtx_inv(
                ch.tensor(g, device="cpu"), xtx, progress=progress
            )
            features = x_xtx_inv.pin_memory()

            target_task = progress.add_task(
                f"Processing targets for {encoder_cfg.name}...",
                total=len(targets),
            )

            for target in targets:
                input_path = str(
                    Path(cfg.output_dir)
                    / cfg.experiments[0].name
                    / encoder_cfg.name
                    / target
                )
                dataset_target = ds.dataset(input_path, format="parquet")
                batch_size = 16384
                scanner = dataset_target.scanner(
                    columns=["grads", "uid"], batch_size=batch_size
                )
                grads_list = []
                uids_list = []

                batch_task = progress.add_task(
                    f"Loading batches for {target}...",
                    total=dataset_target.count_rows() // batch_size,
                )

                for batch in scanner.to_batches():
                    grads_list.extend(
                        batch.column("grads").to_numpy(zero_copy_only=False)
                    )
                    uids_list.extend(
                        batch.column("uid").to_numpy(zero_copy_only=False)
                    )
                    progress.advance(batch_task)

                progress.remove_task(batch_task)

                g_target = np.stack(grads_list)
                uids_target = np.stack(uids_list)
                id_indices = get_indices(
                    target, id=False
                )  # get downstream indices
                mask = np.isin(uids_target, id_indices)
                g_target_pt = ch.tensor(
                    g_target[mask], device="cpu"
                ).pin_memory()

                batch_size = 8192 * 2
                scores = []

                score_task = progress.add_task(
                    f"Computing scores for {target}...",
                    total=len(features) // batch_size + 1,
                )

                for i in range(0, len(features), batch_size):
                    batch = features[i : i + batch_size].cuda()
                    batch_scores = ch.mean(
                        batch @ g_target_pt.cuda().T, axis=1
                    )
                    scores.append(batch_scores.cpu())
                    progress.advance(score_task)

                progress.remove_task(score_task)
                scores = ch.cat(scores)
                avg_scores[target] += scores
                # all_scores[target] = scores.cpu()
                progress.advance(target_task)

            progress.remove_task(target_task)
            progress.advance(encoder_task)

    avg_out_to_loss /= len(cfg.experiments[0].encoders)
    avg_scores = {
        k: v / len(cfg.experiments[0].encoders) for k, v in avg_scores.items()
    }
    final_scores = {
        k: (v * avg_out_to_loss).numpy() for k, v in avg_scores.items()
    }

    root = zarr.open("/raid/pdpl/trak_scores.zarr", mode="a")
    exp_group = root.require_group(experiment_cfg.name)
    for target in targets:
        target_group = exp_group.require_group(target)
        arr = target_group.create_array(
            name="ood_scores",
            dtype=final_scores[target].dtype,
            shape=final_scores[target].shape,
            overwrite=True,
        )
        arr[:] = final_scores[target]
        arr = target_group.create_array(
            name="ood_uids",
            dtype=uids.dtype,
            shape=uids.shape,
            overwrite=True,
        )
        arr[:] = uids


def main():
    cfg = Config()
    cfg.experiments = [ExperimentConfig(name="raw")]
    pprint(cfg)
    for experiment_cfg in cfg.experiments:
        if experiment_cfg.id_dataset_name is not None:
            featurize_with_id(cfg, experiment_cfg)
        else:
            featurize_no_id(cfg, experiment_cfg)


if __name__ == "__main__":
    main()
