import torch

from diffsci.models.karras.ema import ModelEMA
from diffsci.models.karras.karrasmodule_new import (
    EnsembleKarrasModule,
    EnsembleKarrasModuleConfig,
)


class TinyDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x, cond_noise, y=None):
        return self.conv(x)


def _first_parameter(module):
    return next(module.parameters())


def test_traditional_ema_updates_and_restores_weights():
    model = torch.nn.Linear(2, 1, bias=False)
    _first_parameter(model).data.fill_(0.0)
    ema = ModelEMA(model, ema_type="traditional", decay=0.5)

    _first_parameter(model).data.fill_(2.0)
    ema.update(model)
    assert torch.allclose(
        ema.selected_profile()["params"]["weight"],
        torch.ones_like(_first_parameter(model)),
    )

    original = _first_parameter(model).detach().clone()
    backup = ema.apply_to(model)
    assert torch.allclose(_first_parameter(model), torch.ones_like(original))
    ema.restore(model, backup)
    assert torch.allclose(_first_parameter(model), original)


def test_power_ema_first_update_copies_current_weights():
    model = torch.nn.Linear(2, 1, bias=False)
    _first_parameter(model).data.fill_(0.0)
    ema = ModelEMA(model, ema_type="power", power_function_stds=[0.05])

    _first_parameter(model).data.fill_(3.0)
    ema.update(model)
    assert torch.allclose(
        ema.selected_profile()["params"]["weight"],
        torch.full_like(_first_parameter(model), 3.0),
    )
    assert ema.last_beta == 0.0


def test_ensemble_karras_module_ema_checkpoint_and_validation_scope():
    torch.manual_seed(0)
    model = TinyDenoiser()
    config = EnsembleKarrasModuleConfig.from_edm(
        ema_enabled=True,
        ema_type="traditional",
        ema_decay=0.5,
    )
    module = EnsembleKarrasModule(model, config)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    module.set_optimizer_and_scheduler(optimizer)

    x = torch.randn(2, 1, 4, 4)
    sigma = config.noisesampler.sample(x.shape[0]).to(x)
    loss = module.loss_fn(x, sigma)
    loss.backward()
    optimizer.step()
    module.on_before_zero_grad(optimizer)

    assert module.ema_tracker.num_updates == 1
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    assert "model_ema" in checkpoint

    reloaded = EnsembleKarrasModule(TinyDenoiser(), config)
    reloaded.on_load_checkpoint(checkpoint)
    assert reloaded.ema_tracker.num_updates == 1

    original = _first_parameter(module.model).detach().clone()
    module.on_validation_epoch_start()
    assert not torch.allclose(_first_parameter(module.model), original)
    module.on_validation_epoch_end()
    assert torch.allclose(_first_parameter(module.model), original)


def test_ensemble_karras_module_without_ema_keeps_training_loss_working():
    torch.manual_seed(0)
    model = TinyDenoiser()
    config = EnsembleKarrasModuleConfig.from_edm()
    module = EnsembleKarrasModule(model, config)

    x = torch.randn(2, 1, 4, 4)
    sigma = config.noisesampler.sample(x.shape[0]).to(x)
    loss = module.loss_fn(x, sigma)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert module.ema_tracker is None
