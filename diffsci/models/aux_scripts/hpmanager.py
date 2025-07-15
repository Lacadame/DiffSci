import wandb


class HyperparameterManager:
    def __init__(self):
        self.hparams = {}

    def add_model_config(self, model):
        """Extract and add model hyperparameters"""
        if hasattr(model, 'config') and hasattr(model.config, 'export_description'):
            model_hparams = model.config.export_description()
            self.hparams.update({f"model/{k}": v for k, v in model_hparams.items()})

    def add_optimizer_config(self, optimizer_config):
        """Add optimizer configuration"""
        self.hparams.update({f"optimizer_config/{k}": v for k, v in optimizer_config.items()})

    def add_training_config(self, **kwargs):
        """Add training-specific hyperparameters"""
        self.hparams.update({f"training/{k}": v for k, v in kwargs.items()})

    def add_runtime_optimizer_info(self, optimizer, scheduler=None):
        """Add actual optimizer state after creation"""
        # Extract from actual optimizer object
        for group_idx, param_group in enumerate(optimizer.param_groups):
            prefix = f"optimizer_runtime/group_{group_idx}" if len(optimizer.param_groups) > 1 else "optimizer_runtime"
            for key, value in param_group.items():
                if key != 'params':
                    self.hparams[f"{prefix}/{key}"] = value

        self.hparams["optimizer_runtime/type"] = optimizer.__class__.__name__

        if scheduler:
            self.hparams["scheduler_runtime/type"] = scheduler.__class__.__name__
            scheduler_state = scheduler.state_dict()
            for key, value in scheduler_state.items():
                if not key.startswith('_') and isinstance(value, (int, float, str, bool)):
                    self.hparams[f"scheduler_runtime/{key}"] = value

    def log_to_wandb(self):
        """Log all collected hyperparameters to wandb"""
        if wandb.run is not None:
            wandb.config.update(self.hparams)
        else:
            print("Warning: No active wandb run")

    def export_dict(self):
        """Export as dictionary for other logging systems"""
        return self.hparams.copy()
