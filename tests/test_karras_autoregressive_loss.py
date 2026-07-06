import torch

import diffsci.models
from diffsci.models.karras.karrasmodule_new import (
    EnsembleKarrasModule,
    EnsembleKarrasModuleConfig,
)


class ConditionalZeroModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x, t, y=None):
        return torch.zeros_like(x) + self.dummy * x


def _patch_sample(module):
    sample_conditions = []

    def fake_sample(nsamples, shape, y=None, **kwargs):
        sample_conditions.append(y["y"].detach().clone())
        value = float(len(sample_conditions))
        return torch.full((nsamples, *shape), value, device=module.device)

    module.sample = fake_sample
    return sample_conditions


def test_karras_autoregressive_loss_updates_conditioning():
    config = diffsci.models.KarrasModuleConfig.from_edm(
        autoregressive_loss_steps=3,
        autoregressive_loss_diffusion_steps=1,
    )
    module = diffsci.models.KarrasModule(
        ConditionalZeroModel(),
        config,
        conditional=True,
    )
    sample_conditions = _patch_sample(module)

    x = torch.randn(2, 6, 4, 4)
    y = {"y": torch.zeros(2, 4, 4, 4)}
    mask = torch.zeros(2, 1, 4, 4)

    loss = module.autoregressive_loss_fn(x, y, mask)

    assert loss.ndim == 0
    assert len(sample_conditions) == 4
    assert torch.equal(y["y"], torch.zeros_like(y["y"]))
    assert torch.all(sample_conditions[2][-2:] == 1.0)
    assert torch.all(sample_conditions[3][-2:] == 2.0)


def test_ensemble_karras_autoregressive_loss_updates_conditioning():
    config = EnsembleKarrasModuleConfig.from_edm(
        autoregressive_loss_steps=2,
        autoregressive_loss_diffusion_steps=1,
    )
    config.ensemble_size_train = 2
    module = EnsembleKarrasModule(
        ConditionalZeroModel(),
        config,
        conditional=True,
    )
    module.set_loss_metric()
    sample_conditions = _patch_sample(module)

    x = torch.randn(2, 4, 4, 4)
    y = {"y": torch.zeros(2, 4, 4, 4)}

    loss = module.autoregressive_loss_fn(
        x,
        y,
        n_ensemble=config.ensemble_size_train,
    )

    assert loss.ndim == 0
    assert len(sample_conditions) == 2
    assert all(condition.shape == (4, 4, 4) for condition in sample_conditions)


def test_ensemble_karras_default_training_step_uses_original_loss_path():
    config = EnsembleKarrasModuleConfig.from_edm()
    module = EnsembleKarrasModule(
        ConditionalZeroModel(),
        config,
        conditional=True,
    )
    module.log = lambda *args, **kwargs: None

    calls = []

    def fake_loss_fn(x, sigma, y, mask=None, n_ensemble=1):
        calls.append({
            "x_shape": tuple(x.shape),
            "sigma_shape": tuple(sigma.shape),
            "y_shape": tuple(y["y"].shape),
            "n_ensemble": n_ensemble,
        })
        return x.new_tensor(2.0)

    def forbidden_autoregressive_loss(*args, **kwargs):
        raise AssertionError("autoregressive_loss_fn should not be called")

    module.loss_fn = fake_loss_fn
    module.autoregressive_loss_fn = forbidden_autoregressive_loss

    x = torch.randn(2, 3, 4, 4)
    y = {"y": torch.randn(2, 4, 4, 4)}

    loss = module.training_step((x, y), 0)

    assert config.autoregressive_loss_steps == 1
    assert config.autoregressive_loss_diffusion_steps == 100
    assert not module.has_autoregressive_loss()
    assert loss.item() == 2.0
    assert calls == [{
        "x_shape": (2, 3, 4, 4),
        "sigma_shape": (2,),
        "y_shape": (2, 4, 4, 4),
        "n_ensemble": config.ensemble_size_train,
    }]


def test_ensemble_karras_autoregressive_training_step_logs_horizon_losses():
    config = EnsembleKarrasModuleConfig.from_edm(
        autoregressive_loss_steps=3,
        autoregressive_loss_diffusion_steps=1,
        autoregressive_loss_weights=[1.0, 2.0, 1.0],
    )
    module = EnsembleKarrasModule(
        ConditionalZeroModel(),
        config,
        conditional=True,
    )
    _patch_sample(module)

    log_calls = []
    step_losses = [1.0, 2.0, 3.0]

    def fake_step_loss(x, sigma, y, mask=None, n_ensemble=1):
        return x.new_tensor(step_losses.pop(0))

    def fake_log(name, value, **kwargs):
        log_calls.append((name, float(value.detach().cpu())))

    module._loss_fn_for_autoregressive_step = fake_step_loss
    module.log = fake_log

    x = torch.randn(2, 6, 4, 4)
    y = {"y": torch.zeros(2, 4, 4, 4)}

    loss = module.training_step((x, y), 0)

    assert loss.item() == 2.0
    assert [float(v.cpu()) for v in module.last_autoregressive_step_losses] == [
        1.0,
        2.0,
        3.0,
    ]
    assert ("train_ar_loss_horizon_1", 1.0) in log_calls
    assert ("train_ar_loss_horizon_2", 2.0) in log_calls
    assert ("train_ar_loss_horizon_3", 3.0) in log_calls
    assert ("train_loss", 2.0) in log_calls
