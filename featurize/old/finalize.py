# we now need to read the grads back into the proper TRAKer and continue

from pathlib import Path
from pprint import pprint

import pyarrow.dataset as ds
from config.config import Config
from model import Model
from trak import TRAKer

cfg = Config()
# cfg.device="cpu"
pprint(cfg)

encoder_cfg = cfg.encoders[0]
input_path = str(
    Path(cfg.output_dir) / encoder_cfg.name / encoder_cfg.ood_dataset_name
)
dataset = ds.dataset(input_path, format="parquet")
train_set_size = dataset.count_rows()

encoder_cfg = cfg.encoders[0]
model = Model(cfg.encoders[0], "cpu")
model, _, _, _ = model.create_model_and_transforms()
traker = TRAKer(
    save_dir=cfg.save_dir,
    model=model,
    task="clip",
    train_set_size=train_set_size,
    device=cfg.device,
    proj_dim=cfg.proj_dim,
    use_half_precision=True,
)

traker.finalize_features(model_ids=[0, 1])
