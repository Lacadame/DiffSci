# Callbacks originally from diffsci.models.karras.callbacks 

import lightning.pytorch.callbacks as pl_callbacks
import torch


class EMACallback(pl_callbacks.StochasticWeightAveraging):
    def __init__(self, decay=0.99):
        super().__init__(decay)
        self.decay = decay

    def avg_fn(self,
               averaged_model_parameter: torch.Tensor,
               model_parameter: torch.Tensor,
               num_averaged: torch.LongTensor) -> torch.FloatTensor:
        e = averaged_model_parameter
        m = model_parameter
        return self.decay * e + (1. - self.decay) * m


class ScheduleFreeCallback(pl_callbacks.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.model.train()
        if hasattr(pl_module.optimizer, "train"):
            pl_module.optimizer.train()

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.model.eval()
        if hasattr(pl_module.optimizer, "eval"):
            pl_module.optimizer.eval()

    def on_test_epoch_start(self, trainer, pl_module):
        pl_module.model.eval()
        if hasattr(pl_module.optimizer, "eval"):
            pl_module.optimizer.eval()


class NanToZeroGradCallback(pl_callbacks.Callback):
    def on_before_optimizer_step(self,
                                 trainer,
                                 pl_module,
                                 optimizer):
        for p in pl_module.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)