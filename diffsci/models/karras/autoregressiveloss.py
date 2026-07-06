from copy import copy
from typing import Any, Optional

import torch


class AutoregressiveLossMixin:
    """Shared autoregressive training-loss helpers for Karras modules."""

    def has_autoregressive_loss(self) -> bool:
        return getattr(self.config, "autoregressive_loss_steps", 1) > 1

    def autoregressive_loss_fn(
            self,
            x: torch.Tensor,
            y: Optional[Any] = None,
            mask: Optional[torch.Tensor] = None,
            n_ensemble: int = 1,
            nsteps: Optional[int] = None) -> torch.Tensor:
        steps = self._get_autoregressive_loss_steps(nsteps)
        targets = self._split_autoregressive_targets(x, steps)
        masks = self._split_autoregressive_masks(mask, steps, targets)
        weights = self._autoregressive_step_weights(steps, x)

        current_y = self._clone_conditioning(y)
        total_loss = x.new_tensor(0.0)
        self.last_autoregressive_step_losses = []
        self.last_autoregressive_weighted_step_losses = []

        for step, target in enumerate(targets):
            sigma = self.config.noisesampler.sample(target.shape[0]).to(target)
            step_loss = self._loss_fn_for_autoregressive_step(
                target,
                sigma,
                current_y,
                masks[step],
                n_ensemble=n_ensemble
            )
            weighted_step_loss = weights[step] * step_loss
            self.last_autoregressive_step_losses.append(step_loss.detach())
            self.last_autoregressive_weighted_step_losses.append(
                weighted_step_loss.detach()
            )
            total_loss = total_loss + weighted_step_loss

            if step < steps - 1:
                prediction = self._sample_next_autoregressive_condition(
                    target,
                    current_y
                )
                current_y = self._append_autoregressive_prediction(
                    current_y,
                    prediction
                )

        return total_loss

    def log_autoregressive_step_losses(self, prefix: str) -> None:
        for step, step_loss in enumerate(
                getattr(self, "last_autoregressive_step_losses", []),
                start=1):
            self.log(
                f"{prefix}_ar_loss_horizon_{step}",
                step_loss,
                prog_bar=False,
                sync_dist=True
            )

    def _get_autoregressive_loss_steps(self, nsteps: Optional[int]) -> int:
        steps = self.config.autoregressive_loss_steps if nsteps is None else nsteps
        steps = int(steps)
        if steps < 1:
            raise ValueError("autoregressive_loss_steps must be >= 1")
        return steps

    def _split_autoregressive_targets(
            self,
            x: torch.Tensor,
            steps: int) -> list[torch.Tensor]:
        if steps == 1:
            return [x]

        if x.ndim >= 5 and x.shape[1] == steps:
            return [x[:, step] for step in range(steps)]

        if x.ndim >= 4 and x.shape[1] % steps == 0:
            channels_per_step = x.shape[1] // steps
            return list(torch.split(x, channels_per_step, dim=1))

        raise ValueError(
            "Could not split x into autoregressive targets. Expected either "
            "[batch, steps, channels, ...] with steps matching "
            "autoregressive_loss_steps, or a channel-flattened tensor "
            "[batch, steps * channels, ...]."
        )

    def _split_autoregressive_masks(
            self,
            mask: Optional[torch.Tensor],
            steps: int,
            targets: list[torch.Tensor]) -> list[Optional[torch.Tensor]]:
        if mask is None or steps == 1:
            return [mask] * steps

        if mask.ndim >= 5 and mask.shape[1] == steps:
            return [mask[:, step] for step in range(steps)]

        target_channels = targets[0].shape[1]
        if mask.ndim >= 4 and mask.shape[1] == steps * target_channels:
            return list(torch.split(mask, target_channels, dim=1))

        return [mask] * steps

    def _autoregressive_step_weights(
            self,
            steps: int,
            reference: torch.Tensor) -> torch.Tensor:
        weights = getattr(self.config, "autoregressive_loss_weights", None)
        if weights is None:
            weights = torch.ones(steps, device=reference.device,
                                 dtype=reference.dtype)
        else:
            weights = torch.as_tensor(weights, device=reference.device,
                                      dtype=reference.dtype)
            if weights.numel() != steps:
                raise ValueError(
                    "autoregressive_loss_weights must have one value per "
                    "autoregressive loss step"
                )

        return weights / weights.sum().clamp(min=torch.finfo(weights.dtype).eps)

    def _sample_next_autoregressive_condition(
            self,
            target: torch.Tensor,
            y: Optional[Any]) -> torch.Tensor:
        if y is None:
            raise ValueError(
                "Autoregressive loss requires conditional data so generated "
                "predictions can be fed back into y['y']."
            )

        shape = list(target.shape[1:])
        nsamples = target.shape[0]
        sample_kwargs = dict(
            shape=shape,
            y=y,
            guidance=getattr(self.config, "autoregressive_loss_guidance", 1.0),
            nsteps=getattr(self.config, "autoregressive_loss_diffusion_steps", 100),
            record_history=False,
            maximum_batch_size=getattr(
                self.config,
                "autoregressive_loss_maximum_batch_size",
                None
            ),
            integrator=getattr(
                self.config,
                "autoregressive_loss_integrator",
                None
            ),
            return_in_latent_space=False
        )

        if self._conditioning_has_batch_dimension(y, nsamples):
            samples = [
                self.sample(
                    nsamples=1,
                    **{**sample_kwargs,
                       "y": self._select_conditioning_item(y, i, nsamples)}
                )[0]
                for i in range(nsamples)
            ]
            return torch.stack(samples, dim=0).to(target)

        return self.sample(nsamples=nsamples, **sample_kwargs).to(target)

    def _append_autoregressive_prediction(
            self,
            y: Optional[Any],
            prediction: torch.Tensor) -> Optional[Any]:
        if not isinstance(y, dict) or "y" not in y:
            raise ValueError(
                "Autoregressive loss expects y to be a dict containing key 'y'."
            )

        updated = self._clone_conditioning(y)
        y_tensor = updated["y"]
        prediction = prediction.detach().to(y_tensor)

        if y_tensor.ndim == prediction.ndim - 1:
            if prediction.shape[0] != 1:
                raise ValueError(
                    "Cannot append batched predictions to unbatched y['y']."
                )
            prediction = prediction[0]

        if y_tensor.ndim != prediction.ndim:
            raise ValueError(
                f"Prediction rank {prediction.ndim} is incompatible with "
                f"y['y'] rank {y_tensor.ndim}."
            )

        channel_dim = 1 if y_tensor.ndim >= 4 else 0
        channels_per_step = prediction.shape[channel_dim]
        if y_tensor.shape[channel_dim] < channels_per_step:
            raise ValueError(
                "y['y'] has fewer channels than the generated prediction."
            )

        updated["y"] = torch.cat(
            [
                y_tensor.narrow(
                    channel_dim,
                    channels_per_step,
                    y_tensor.shape[channel_dim] - channels_per_step
                ),
                prediction
            ],
            dim=channel_dim
        )
        return updated

    def _conditioning_has_batch_dimension(self, y: Any, batch_size: int) -> bool:
        if isinstance(y, dict):
            if isinstance(y.get("y"), torch.Tensor):
                y_tensor = y["y"]
                return y_tensor.ndim >= 4 and y_tensor.shape[0] == batch_size
            for value in y.values():
                if self._tensor_has_batch_dimension(value, batch_size):
                    return True
        return self._tensor_has_batch_dimension(y, batch_size)

    def _tensor_has_batch_dimension(self, value: Any, batch_size: int) -> bool:
        return (isinstance(value, torch.Tensor) and value.ndim > 0 and
                value.shape[0] == batch_size)

    def _select_conditioning_item(
            self,
            value: Any,
            index: int,
            batch_size: int) -> Any:
        if isinstance(value, dict):
            return {
                key: self._select_conditioning_item(val, index, batch_size)
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [
                self._select_conditioning_item(val, index, batch_size)
                for val in value
            ]
        if isinstance(value, tuple):
            return tuple(
                self._select_conditioning_item(val, index, batch_size)
                for val in value
            )
        if self._tensor_has_batch_dimension(value, batch_size):
            return value[index]
        return value

    def _clone_conditioning(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._clone_conditioning(val)
                    for key, val in value.items()}
        if isinstance(value, list):
            return [self._clone_conditioning(val) for val in value]
        if isinstance(value, tuple):
            return tuple(self._clone_conditioning(val) for val in value)
        if isinstance(value, torch.Tensor):
            return value.clone()
        try:
            return copy(value)
        except TypeError:
            return value
