from open_clip import create_model_and_transforms, get_tokenizer


class Model:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

    def create_model_and_transforms(self):
        if self.cfg.path:  # Check if it's a local checkpoint
            model, preprocess_train, preprocess_val = (
                create_model_and_transforms(
                    self.cfg.architecture,
                    precision=self.cfg.precision,
                    pretrained=self.cfg.path,
                    load_weights_only=False,
                )
            )
            tokenizer = get_tokenizer(self.cfg.architecture)
        else:  # Assume it's a HuggingFace hub model
            model, preprocess_train, preprocess_val = (
                create_model_and_transforms(
                    f"hf-hub:{self.cfg.url}",
                    precision=self.cfg.precision,
                    load_weights_only=False,
                )
            )
            tokenizer = get_tokenizer(f"hf-hub:{self.cfg.url}")

        model = model.to(self.device)
        return model, tokenizer, preprocess_train, preprocess_val
