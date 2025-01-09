from pathlib import Path

import pyarrow.dataset as ds
import torch
from trak.traker import TRAKer


class PreComputedGradients(TRAKer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loaded_data = None

    def load_parquet_data(self, base_path: str):
        """Load all parquet files from the given path."""
        base_path = Path(base_path)
        dataset = ds.dataset(base_path, format="parquet")

        # Load all data into memory
        table = dataset.to_table()
        self._loaded_data = {
            "uid": table["uid"].to_numpy(),
            # grads are already projected and normalized
            "grads": torch.tensor(
                table["grads"].to_numpy().tolist(),
                dtype=self.dtype,
                device=self.device,
            ),
            "loss_grads": torch.tensor(
                table["loss_grads"].to_numpy(),
                dtype=self.dtype,
                device=self.device,
            ),
        }
    

    def featurize(self, batch_indices, inds=None, num_samples=None):
        """Process pre-computed gradients that are already projected and normalized."""
        assert (
            self._loaded_data is not None
        ), "Must call load_parquet_data before featurizing"
        assert (inds is None) or (
            num_samples is None
        ), "Exactly one of num_samples and inds should be specified"

        if num_samples is not None:
            inds = range(self._last_ind, self._last_ind + num_samples)
            self._last_ind += num_samples

        # Get the pre-computed gradients for this batch
        grads = self._loaded_data["grads"][batch_indices]
        loss_grads = self._loaded_data["loss_grads"][batch_indices]

        # Store the results directly since they're already projected and normalized
        self.saver.current_store["grads"][inds] = (
            grads.to(self.dtype).cpu().clone().detach()
        )
        self.saver.current_store["out_to_loss"][inds] = (
            loss_grads.to(self.dtype).cpu().clone().detach()
        )
        self.saver.current_store["is_featurized"][inds] = 1
        self.saver.serialize_current_model_id_metadata()
