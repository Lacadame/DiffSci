import torch

from diffsci.models.karras.karrasmodule_new import (
    EnsembleKarrasModule,
    EnsembleKarrasModuleConfig,
)


class DummyDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conditional_embedding = torch.nn.Module()
        self.conditional_embedding.bat_emb = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, kernel_size=1),
            torch.nn.Flatten(),
            torch.nn.Linear(2, 2),
        )
        self.conditional_embedding.geo_emb = torch.nn.Linear(2, 2)
        self.head = torch.nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, x, t=None, y=None):
        return self.head(x)


def optimizer_parameter_ids(module):
    return {
        id(param)
        for group in module.optimizer.param_groups
        for param in group["params"]
    }


def test_freezes_bathymetry_module_and_filters_optimizer():
    config = EnsembleKarrasModuleConfig.from_edm(
        freeze_layer_patterns=["conditional_embedding.bat_emb"],
        freeze_layer_strict=True,
    )
    module = EnsembleKarrasModule(DummyDenoiser(), config)

    named_parameters = dict(module.model.named_parameters())
    frozen_parameters = {
        name: param
        for name, param in named_parameters.items()
        if name.startswith("conditional_embedding.bat_emb.")
    }
    trainable_parameters = {
        name: param
        for name, param in named_parameters.items()
        if not name.startswith("conditional_embedding.bat_emb.")
    }

    assert frozen_parameters
    assert module.frozen_parameter_count == sum(
        param.numel() for param in frozen_parameters.values()
    )
    assert all(not param.requires_grad for param in frozen_parameters.values())
    assert all(param.requires_grad for param in trainable_parameters.values())

    optimizer_ids = optimizer_parameter_ids(module)
    assert all(id(param) not in optimizer_ids
               for param in frozen_parameters.values())
    assert all(id(param) in optimizer_ids
               for param in trainable_parameters.values())


def test_accepts_model_prefix_and_parameter_globs():
    config = EnsembleKarrasModuleConfig.from_edm(
        freeze_layer_patterns=["model.conditional_embedding.bat_emb.*weight"],
        freeze_layer_strict=True,
    )
    module = EnsembleKarrasModule(DummyDenoiser(), config)

    frozen_names = {
        name for name, param in module.model.named_parameters()
        if not param.requires_grad
    }

    assert frozen_names == {
        "conditional_embedding.bat_emb.0.weight",
        "conditional_embedding.bat_emb.2.weight",
    }


def test_strict_mode_rejects_unmatched_patterns():
    config = EnsembleKarrasModuleConfig.from_edm(
        freeze_layer_patterns=["does_not_exist"],
        freeze_layer_strict=True,
    )

    try:
        EnsembleKarrasModule(DummyDenoiser(), config)
    except ValueError as exc:
        assert "does_not_exist" in str(exc)
    else:
        raise AssertionError("Expected unmatched freeze pattern to fail")


def test_config_export_preserves_freeze_settings():
    config = EnsembleKarrasModuleConfig.from_edm(
        freeze_layer_patterns=["conditional_embedding.bat_emb"],
        freeze_layer_strict=False,
    )
    loaded = EnsembleKarrasModuleConfig.load_from_description_with_tag(
        config.export_description()
    )

    assert loaded.freeze_layer_patterns == ["conditional_embedding.bat_emb"]
    assert loaded.freeze_layer_strict is False
