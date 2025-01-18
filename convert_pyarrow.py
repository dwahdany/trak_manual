from pathlib import Path
from pprint import pprint

import numpy as np
import pyarrow.dataset as ds
import zarr
from rich.progress import Progress

from config.config import Config

cfg = Config()
pprint(cfg)


def convert_column(name: str, data):
    match name:
        case "grads":
            return np.stack(data[name].to_numpy())
        case "uid":
            uids = data[name].to_numpy()
            processed_uids = np.array(
                [(int(uid[:16], 16), int(uid[16:32], 16)) for uid in uids],
                np.dtype("u8,u8"),
            )
            return processed_uids
        case _:
            return data[name].to_numpy()


column_to_array = convert_column
with Progress() as progress:
    encoder_task = progress.add_task(
        "Processing encoders", total=len(cfg.encoders)
    )

    for encoder_cfg in cfg.encoders:
        input_path = str(
            Path(cfg.output_dir)
            / encoder_cfg.name
            / encoder_cfg.ood_dataset_name
        )
        dataset = ds.dataset(input_path, format="parquet")
        # Load all data into memory
        table = dataset.to_table()

        # Create zarr store
        store_path = Path(input_path) / "data.zarr"
        store = zarr.DirectoryStore(str(store_path))
        root = zarr.group(store=store)

        column_task = progress.add_task(
            f"Converting columns for {encoder_cfg.name}",
            total=len(table.column_names),
        )

        for col in table.column_names:
            root.create_dataset(
                col, data=convert_column(col, table), chunks=True
            )
            progress.advance(column_task)

        progress.remove_task(column_task)
        progress.advance(encoder_task)

        print(f"Saved zarr store to {store_path}")
        print("Store info:")
        print(root.tree())
