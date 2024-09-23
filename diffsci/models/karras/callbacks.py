import lightning.pytorch.callbacks as pl_callbacks

from .edmbatchnorm import EDMBatchNorm


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


class EDMBatchNormCallback(pl_callbacks.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if (hasattr(pl_module, "edm_batch_norm") and
                isinstance(pl_module.edm_batch_norm, EDMBatchNorm)):
            pl_module.edm_batch_norm.use_running_mean = True


class NanToZeroGradCallback(pl_callbacks.Callback):
    def on_before_optimizer_step(self,
                                 trainer,
                                 pl_module,
                                 optimizer):
        for p in pl_module.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
