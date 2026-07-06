import os
import pathlib
import sys
import tempfile

import torch
import yaml


DIFFSCI_ROOT = pathlib.Path(__file__).resolve().parents[1]
OCEANGEN_ROOT = pathlib.Path("/home/ubuntu/repos/OceanGen")
CONFIG_PATH = OCEANGEN_ROOT / (
    "configs/ldm/"
    "[ocean]-[GUIANA]-[20250801]-[D=4]-[Skip=96]-[r_lat=True]-"
    "[GuianaEncoder]-[att-on]-[Fine-Tunned]-[GT=analysis].yaml"
)


def _prepare_imports():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    for path in [str(DIFFSCI_ROOT), str(OCEANGEN_ROOT)]:
        if path not in sys.path:
            sys.path.insert(0, path)


def _write_temp_config() -> pathlib.Path:
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    config["device"] = "cpu"
    config["model_name"] = config["model_name"] + "-EMA-SMOKE"
    config["output"]["folder"] = "/tmp/oceangen_ema_smoke"
    config["data"]["batch_size"] = 1
    config["data"]["num_workers"] = 0
    config["data"]["dataloader_args"]["device"] = "cpu"
    config["training"]["devices"] = 1
    config["training"]["accelerator"] = "cpu"
    config["training"]["strategy"] = "auto"
    config["training"]["precision"] = 32
    config["training"]["fast_dev_run"] = True

    karras_config = config["training"].setdefault("karras_config", {})
    karras_config["autoregressive_loss_steps"] = 4
    karras_config["autoregressive_loss_diffusion_steps"] = 4
    karras_config["ema_enabled"] = True
    karras_config["ema_type"] = "power"
    karras_config["ema_power_function_stds"] = [0.05]
    karras_config["ema_use_for_validation"] = True
    karras_config["ema_use_for_sampling"] = True

    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="oceangen_ema_smoke_"))
    config_path = tmpdir / "config.yaml"
    with open(config_path, "w") as file:
        yaml.safe_dump(config, file, default_flow_style=False)
    return config_path


def main():
    _prepare_imports()

    from oceangen.trainers import ModelLoaderTrainer
    from diffsci.models.karras.karrasmodule_new import EnsembleKarrasModule

    config_path = _write_temp_config()
    trainer = ModelLoaderTrainer(configpath=config_path)
    trainer.module.log = lambda *args, **kwargs: None

    assert isinstance(trainer.module, EnsembleKarrasModule)
    assert trainer.module.config.ema_enabled is True
    assert trainer.module.config.ema_type == "power"
    assert trainer.module.config.autoregressive_loss_steps == 4

    train_dataloader, _ = trainer.load_train_dataloader()
    batch = next(iter(train_dataloader))
    x, y = batch
    assert tuple(x.shape[:2]) == (1, 12)
    assert tuple(y["y"].shape[:2]) == (1, 12)

    trainer.module.train()
    trainer.module.optimizer.zero_grad(set_to_none=True)
    loss = trainer.module.training_step(batch, 0)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    trainer.module.optimizer.step()
    trainer.module.on_before_zero_grad(trainer.module.optimizer)
    assert trainer.module.ema_tracker.num_updates == 1

    checkpoint = {}
    trainer.module.on_save_checkpoint(checkpoint)
    assert "model_ema" in checkpoint

    print("config_path:", config_path)
    print("loss:", float(loss.detach().cpu()))
    print("ema_updates:", trainer.module.ema_tracker.num_updates)
    print("ema_type:", trainer.module.config.ema_type)
    print("batch_x_shape:", tuple(x.shape))
    print("batch_y_shape:", tuple(y["y"].shape))


if __name__ == "__main__":
    main()
