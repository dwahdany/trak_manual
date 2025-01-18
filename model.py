import tempfile
from pathlib import Path
from typing import Optional

from cloudpathlib import S3Client
from open_clip import create_model_and_transforms, get_tokenizer


class Model:
    def __init__(self, cfg, device, s3_endpoint_url: Optional[str] = None):
        self.cfg = cfg
        self.device = device
        if s3_endpoint_url:
            self.s3_client = S3Client(endpoint_url=s3_endpoint_url)

    def _get_local_weights_path(self, path: str) -> Path:
        """Downloads S3 weights to a temporary file if needed."""
        if path.startswith("s3://"):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
            print(f"Downloading weights from {path} to {tmp.name}")
            s3_path = self.s3_client.S3Path(path)
            s3_path.download_to(tmp.name)
            return Path(tmp.name)
        return Path(path)

    def create_model_and_transforms(self):
        if path := self.cfg.get("path"):  # Check if it's a local checkpoint
            path = self._get_local_weights_path(path)
            model, preprocess_train, preprocess_val = (
                create_model_and_transforms(
                    self.cfg["architecture"],
                    precision=self.cfg.precision,
                    pretrained=str(path),
                    load_weights_only=False,
                )
            )
            if path.parts[1] == "tmp":
                path.unlink()
            tokenizer = get_tokenizer(self.cfg["architecture"])
        else:  # Assume it's a HuggingFace hub model
            model, preprocess_train, preprocess_val = (
                create_model_and_transforms(
                    f"hf-hub:{self.cfg['url']}",
                    precision=self.cfg.precision,
                )
            )
            tokenizer = get_tokenizer(f"hf-hub:{self.cfg['url']}")

        model = model.to(self.cfg.device)
        return model, tokenizer, preprocess_train, preprocess_val
