from pathlib import Path
from pprint import pprint

import numpy as np
import pyarrow.dataset as ds
import zarr
from config.config import Config
from rich.progress import Progress

cfg = Config()
pprint(cfg)


def convert_column(name: str, data):
    match name:
        case "grads":
            return np.stack(data[name].to_numpy())
        case "uid":
            uids = data[name].to_numpy()
            # Convert strings to fixed-length S32 dtype
            return np.array(uids, dtype="S32")
        case _:
            return data[name].to_numpy()


column_to_array = convert_column
with Progress() as progress:
    experiment_task = progress.add_task(
        "Processing experiments", total=len(cfg.experiments)
    )
    for experiment_cfg in cfg.experiments:
        encoder_task = progress.add_task(
            "Processing encoders", total=len(experiment_cfg.encoders)
        )

        for encoder_cfg in experiment_cfg.encoders:
            targets = experiment_cfg.target_datasets
            for target in targets:
                input_path = str(
                    Path(cfg.output_dir)
                    / experiment_cfg.name
                    / encoder_cfg.name
                    / target
                )
                store_path = Path(input_path) / "data.zarr"
                done_file = Path(input_path) / "zarr.done"

                # Check if zarr store and done file already exist
                if store_path.exists() and done_file.exists():
                    print(f"Zarr store already exists at {store_path}")
                    print("Store info:")
                    root = zarr.open_group(str(store_path))
                    print(root.tree())
                    progress.advance(encoder_task)
                    continue

                try:
                    parquet_files = list(Path(input_path).glob("*.parquet"))
                    dataset = ds.dataset(parquet_files, format="parquet")
                except Exception as e:
                    print(f"Skipping {encoder_cfg.name} because of error: {e}")
                    continue
                # Load all data into memory
                table = dataset.to_table()

                # Create zarr store
                root = zarr.open(str(store_path), mode="a")

                column_task = progress.add_task(
                    f"Converting columns for {encoder_cfg.name}",
                    total=len(table.column_names),
                )

                # Check for duplicate UIDs
                uids = table["uid"].to_numpy()
                unique_uids = set(uids)
                if len(unique_uids) != len(uids):
                    print(
                        f"WARNING: Found {len(uids) - len(unique_uids)} ({len(uids)} - {len(unique_uids)}) duplicate UIDs in {encoder_cfg.name}"
                    )
                    # Keep only first occurrence of each UID
                    _, unique_indices = np.unique(uids, return_index=True)
                    uids = uids[unique_indices]
                    table = table.take(unique_indices)

                print(f"Got {len(uids)} unique UIDs")
                if "num_samples" in cfg.datasets[target]:
                    if len(uids) != (
                        expected_num_samples := cfg.datasets[
                            target
                        ].num_samples
                    ):
                        print(
                            f"WARNING: Expected {expected_num_samples} UIDs, got {len(uids)}. Skipping."
                        )
                        continue

                for col in table.column_names:
                    data = convert_column(col, table)
                    arr = root.create_array(
                        col,
                        shape=data.shape,
                        dtype=data.dtype,
                        overwrite=True,
                    )
                    arr[:] = data
                    progress.advance(column_task)

                progress.remove_task(column_task)
                progress.advance(encoder_task)

                print(f"Saved zarr store to {store_path}")
                print("Store info:")
                print(root.tree())

                # Create done file to mark completion
                done_file.touch()

        progress.advance(experiment_task)
    progress.remove_task(experiment_task)
