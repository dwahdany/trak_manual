import logging
from functools import partial
from typing import Iterable, Optional, Union

import torch
from trak.gradient_computers import (
    AbstractGradientComputer,
    FunctionalGradientComputer,
)
from trak.modelout_functions import AbstractModelOutput
from trak.projectors import AbstractProjector
from trak.savers import AbstractSaver
from trak.score_computers import AbstractScoreComputer
from trak.traker import TRAKer
from trak.utils import get_num_params

ch = torch
from trak.modelout_functions import TASK_TO_MODELOUT


class Featurizer(TRAKer):
    def __init__(
        self,
        model: torch.nn.Module,
        task: Union[AbstractModelOutput, str],
        model_id: int,
        train_set_size: Optional[int] = None,
        save_dir: str = "./trak_results",
        load_from_save_dir: bool = True,
        device: Union[str, torch.device] = "cuda",
        gradient_computer: AbstractGradientComputer = FunctionalGradientComputer,
        projector: Optional[AbstractProjector] = None,
        saver: Optional[AbstractSaver] = None,
        score_computer: Optional[AbstractScoreComputer] = None,
        proj_dim: int = 2048,
        logging_level=logging.INFO,
        use_half_precision: bool = True,
        proj_max_batch_size: int = 32,
        projector_seed: int = 0,
        grad_wrt: Optional[Iterable[str]] = None,
        lambda_reg: float = 0.0,
    ):
        self.model_id = model_id
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(console_handler)

        self.model = model
        self.task = task
        self.device = device
        self.dtype = ch.float16 if use_half_precision else ch.float32
        # self.dtype = torch.float8_e4m3fn
        self.grad_wrt = grad_wrt

        self.num_params = get_num_params(self.model)
        if self.grad_wrt is not None:
            d = dict(self.model.named_parameters())
            self.num_params_for_grad = sum(
                [d[param_name].numel() for param_name in self.grad_wrt]
            )
        else:
            self.num_params_for_grad = self.num_params
        # inits self.projector
        self.proj_seed = projector_seed
        self.init_projector(
            projector=projector,
            proj_dim=proj_dim,
            proj_max_batch_size=proj_max_batch_size,
        )

        # normalize to make X^TX numerically stable
        # doing this instead of normalizing the projector matrix
        self.normalize_factor = ch.sqrt(
            ch.tensor(self.num_params_for_grad, dtype=ch.float32)
        )

        if type(self.task) is str:
            self.task = TASK_TO_MODELOUT[self.task]()

        self.gradient_computer = gradient_computer(
            model=self.model,
            task=self.task,
            grad_dim=self.num_params_for_grad,
            dtype=self.dtype,
            device=self.device,
            grad_wrt=self.grad_wrt,
        )

    def get_losses(self, image, label):
        all_im_embs = self.task.image_embeddings
        all_txt_embs = self.task.text_embeddings
        N = self.task.num_computed_embeddings
        sim_bs = self.task.sim_batch_size

        clip_inputs = {"image": image.unsqueeze(0), "text": label.unsqueeze(0)}
        image_embeddings, text_embeddings, _ = ch.func.functional_call(
            self.model,
            (
                self.gradient_computer.func_weights,
                self.gradient_computer.func_buffers,
            ),
            args=(),
            kwargs=clip_inputs,
        )
        ii = ch.multinomial(
            input=ch.arange(N).float(), num_samples=sim_bs, replacement=False
        )
        loss_img = -ch.logsumexp(
            -image_embeddings @ (text_embeddings - all_txt_embs[ii]).T, dim=1
        ).squeeze()
        loss_txt = -ch.logsumexp(
            -text_embeddings @ (image_embeddings - all_im_embs[ii]).T, dim=1
        ).squeeze()
        return loss_img, loss_txt

    def featurize(self, batch):
        with ch.amp.autocast(device_type="cuda", dtype=torch.float16):
            with ch.no_grad():
                grads = self.gradient_computer.compute_per_sample_grad(
                    batch=batch
                )
                grads = self.projector.project(grads, model_id=self.model_id)
                grads /= self.normalize_factor
                loss_grads = self.gradient_computer.compute_loss_grad(batch)
                # loss_img, loss_txt = torch.vmap(
                #     partial(self.task.get_output, separate_loss=True),
                #     in_dims=(None, None, None, *([0] * len(batch))),
                #     randomness="different",
                # )(
                #     self.gradient_computer.model,
                #     self.gradient_computer.func_weights,
                #     self.gradient_computer.func_buffers,
                #     *batch,
                # )
                loss_img, loss_txt = torch.vmap(
                    partial(self.get_losses),
                    in_dims=tuple([0] * len(batch)),
                    randomness="different",
                )(
                    *batch,
                )
        return grads, loss_grads, loss_img, loss_txt

    def load_checkpoint(self, *args, **kwargs):
        raise NotImplementedError()

    def finalize_features(self, *args, **kwargs):
        raise NotImplementedError()

    def start_scoring_checkpoint(self, *args, **kwargs):
        raise NotImplementedError()

    def score(self, *args, **kwargs):
        raise NotImplementedError()

    def finalize_scores(self, *args, **kwargs):
        raise NotImplementedError()
