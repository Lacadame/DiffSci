from typing import Any

import torch
from torch import Tensor
from jaxtyping import Float

from porenet.models import KarrasModule, KarrasModuleConfig


# TODO: Make conditional encoding here. I'll assume that the encoding is
# done by the encoder model, and the decoding is done by the model,
# and that there is not conditional decoding.
# This is a simplification that can be relaxed later.

class KarrasEncoder(KarrasModule):
    def __init__(self,
                 model: torch.nn.Module,
                 encoder_model: torch.nn.Module,
                 config: KarrasModuleConfig,
                 masked: bool = False,
                 autoencoder: None | torch.nn.Module = None,
                 autoencoder_conditional: bool = False):
        super().__init__(model=model,
                         config=config,
                         conditional=True,
                         masked=masked,
                         autoencoder=autoencoder,
                         autoencoder_conditional=autoencoder_conditional)
        self.encoder_model = encoder_model

    def export_description(self) -> dict[str, Any]:
        base_description = super().export_description()
        encoder_description = self.encoder_model.export_description()
        return dict(base_description=base_description,
                    encoder_description=encoder_description)

    def loss_fn(self,
                x: Float[Tensor, "batch *shape"],  # noqa: F821
                sigma: Float[Tensor, "batch"],  # noqa: F821
                mask: None | Float[Tensor, "batch *shape"] = None  # noqa: F821
                ) -> Float[Tensor, ""]:  # noqa: F821, F722
        y = self.encoder_model(x)
        loss = super().loss_fn(x, sigma, y, mask)
        return loss

    def training_step(self, batch, batch_idx):
        x, _, mask = self.select_batch(batch)
        sigma = self.config.noisesampler.sample(x.shape[0]).to(x)  # [nbatch]
        loss = self.loss_fn(x, sigma, mask)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, mask = self.select_batch(batch)
        sigma = self.config.noisesampler.sample(x.shape[0]).to(x)  # [nbatch]
        loss = self.loss_fn(x, sigma, mask)
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def select_batch(self, batch):
        # TODO: Kind of hacky. When we make conditional encoding,
        # this should change
        self.conditional = False
        res = super().select_batch(batch)
        self.conditional = True
        return res

