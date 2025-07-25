import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from cloudpathlib import S3Client
from open_clip import create_model_and_transforms, get_tokenizer

if np.lib.NumpyVersion(np.__version__) < "2.0.0":
    import numpy

    numpy._core = numpy.core
    np._core = numpy.core


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
        elif pretrained := self.cfg.get("pretrained"):
            model, preprocess_train, preprocess_val = (
                create_model_and_transforms(
                    self.cfg["architecture"],
                    precision=self.cfg.precision,
                    pretrained=pretrained,
                )
            )
            tokenizer = get_tokenizer(self.cfg["architecture"])
        else:  # Assume it's a HuggingFace hub model
            model, preprocess_train, preprocess_val = (
                create_model_and_transforms(
                    f"hf-hub:{self.cfg['url']}",
                    precision=self.cfg.precision,
                )
            )
            tokenizer = get_tokenizer(f"hf-hub:{self.cfg['url']}")

        model = model.to(self.device)
        return model, tokenizer, preprocess_train, preprocess_val

def give_zs_weights(model, toks):
    weights = []
    for cls_toks in toks:
        class_embeddings = model.encode_text(cls_toks.to("cuda"))
        class_embedding = torch.nn.functional.normalize(class_embeddings, dim=-1).mean(
            dim=0
        )
        class_embedding = class_embedding / class_embedding.norm()
        weights.append(class_embedding)
    return torch.stack(weights, dim=1)

class ZeroshotClassifier(nn.Module):
    def __init__(self, model, zeroshot_templates):
        super().__init__()
        self.model = model
        self.zeroshot_templates = zeroshot_templates # [C, N] templates for C classes and N examples

    def forward(self, image):
        image_embeddings = self.model.encode_image(image) # [B, D]
        text_embeddings = give_zs_weights(self.model, self.zeroshot_templates) # [D, C]
        logits = (image_embeddings @ text_embeddings) # [B, C]
        return logits

class FrozenZeroshotClassifier(nn.Module):
    def __init__(self, model, zeroshot_templates):
        super().__init__()
        self.model = model
        self.zeroshot_templates = zeroshot_templates # [C, N] templates for C classes and N examples
        self.text_embeddings = give_zs_weights(self.model, self.zeroshot_templates) # [D, C]

    def forward(self, image):
        image_embeddings = self.model.encode_image(image) # [B, D]
        logits = (image_embeddings @ self.text_embeddings) # [B, C]
        return logits
