import torch

from diffsci.models.karras.karrasmodule_new import (
    EnsembleKarrasModule,
    EnsembleKarrasModuleConfig,
)


class DummyDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x, t=None, y=None):
        return self.net(x)


def make_module(**config_kwargs):
    config = EnsembleKarrasModuleConfig.from_edm(**config_kwargs)
    module = EnsembleKarrasModule(DummyDenoiser(), config)
    module.log = lambda *args, **kwargs: None
    return module


def test_replay_training_step_combines_losses_with_weight():
    module = make_module(
        replay_enabled=True,
        replay_loss_weight=0.25,
    )

    def fake_loss(batch, **kwargs):
        if batch == "finetune":
            return torch.tensor(2.0)
        if batch == "replay":
            return torch.tensor(4.0)
        raise AssertionError(f"unexpected batch: {batch}")

    module._training_loss_from_batch = fake_loss
    loss = module.training_step(
        {"finetune": "finetune", "replay": "replay"},
        batch_idx=0,
    )

    assert torch.equal(loss, torch.tensor(3.0))


def test_replay_training_step_uses_scheduled_weight():
    module = make_module(
        replay_enabled=True,
        replay_loss_weight=0.25,
        replay_loss_schedule={
            "enabled": True,
            "type": "linear",
            "start_weight": 0.5,
            "end_weight": 0.1,
            "num_epochs": 10,
        },
    )
    module.current_replay_loss_weight = lambda: 0.5

    def fake_loss(batch, **kwargs):
        if batch == "finetune":
            return torch.tensor(2.0)
        if batch == "replay":
            return torch.tensor(4.0)
        raise AssertionError(f"unexpected batch: {batch}")

    module._training_loss_from_batch = fake_loss
    loss = module.training_step(
        {"finetune": "finetune", "replay": "replay"},
        batch_idx=0,
    )

    assert torch.equal(loss, torch.tensor(4.0))


def test_replay_schedule_interpolates_linearly():
    weight = EnsembleKarrasModule._scheduled_replay_loss_weight(
        {
            "enabled": True,
            "type": "linear",
            "start_weight": 0.2,
            "end_weight": 0.05,
            "num_epochs": 10,
        },
        default_weight=0.1,
        position=5,
    )

    assert weight == 0.125


def test_pretrained_weight_regularization_penalizes_weight_drift():
    module = make_module(
        pretrained_weight_regularization={
            "enabled": True,
            "weight": 0.5,
            "include_patterns": ["net.weight"],
            "normalize": False,
        }
    )
    module.initialize_pretrained_weight_regularization_reference()

    with torch.no_grad():
        module.model.net.weight.add_(1.0)

    loss = module.pretrained_weight_regularization_loss()

    assert torch.equal(loss, torch.tensor(0.5))


def test_training_step_adds_pretrained_weight_regularization():
    module = make_module(
        pretrained_weight_regularization={
            "enabled": True,
            "weight": 0.5,
            "include_patterns": ["net.weight"],
        }
    )
    module._training_loss_from_batch = lambda batch, **kwargs: torch.tensor(2.0)
    module.pretrained_weight_regularization_loss = lambda: torch.tensor(0.5)

    loss = module.training_step("finetune", batch_idx=0)

    assert torch.equal(loss, torch.tensor(2.5))


def test_replay_enabled_requires_combined_batch_dict():
    module = make_module(replay_enabled=True)

    try:
        module.training_step(torch.zeros(1, 1, 4, 4), batch_idx=0)
    except ValueError as exc:
        assert "keys 'finetune' and 'replay'" in str(exc)
    else:
        raise AssertionError("Expected replay-enabled training to reject plain batch")


def test_replay_batch_rejected_when_replay_disabled():
    module = make_module(replay_enabled=False)

    try:
        module.training_step({"finetune": "finetune", "replay": "replay"},
                             batch_idx=0)
    except ValueError as exc:
        assert "replay_enabled is False" in str(exc)
    else:
        raise AssertionError("Expected replay batch to fail when replay is disabled")


def test_config_export_preserves_replay_settings():
    replay_loss_schedule = {
        "enabled": True,
        "type": "cosine",
        "start_weight": 0.2,
        "end_weight": 0.05,
        "num_epochs": 20,
    }
    pretrained_weight_regularization = {
        "enabled": True,
        "weight": 1e-4,
        "include_patterns": ["*"],
        "exclude_patterns": ["conditional_embedding.bat_emb"],
    }
    config = EnsembleKarrasModuleConfig.from_edm(
        replay_enabled=True,
        replay_loss_weight=0.2,
        replay_loss_schedule=replay_loss_schedule,
        replay_validation_enabled=True,
        pretrained_weight_regularization=pretrained_weight_regularization,
    )
    loaded = EnsembleKarrasModuleConfig.load_from_description_with_tag(
        config.export_description()
    )

    assert loaded.replay_enabled is True
    assert loaded.replay_loss_weight == 0.2
    assert loaded.replay_loss_schedule == replay_loss_schedule
    assert loaded.replay_validation_enabled is True
    assert loaded.pretrained_weight_regularization == pretrained_weight_regularization
