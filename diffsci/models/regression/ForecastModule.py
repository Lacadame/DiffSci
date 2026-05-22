"""
Forecast Training Module

Lightning-based training module for deterministic ocean forecasting models.
Adapted from the diffusion-based Karras module, but simplified for direct prediction
without noise/diffusion processes.

Features:
- Multi-loss support (MSE, MAE, Huber, custom oceanographic losses)
- Autoencoder support for latent space training
- Conditional embeddings
- Masking support for incomplete data
- Comprehensive metrics and validation
"""

from typing import Callable, Any, Optional, Union, Dict, Tuple, List
import torch
import lightning
from torch import Tensor
from jaxtyping import Float, Bool

__all__ = ['ForecastModuleConfig', 'ForecastModule']


Scaler = Callable[[Float[Tensor, '*shape']],  # noqa: F821
                  Float[Tensor, '*shape']]  # noqa: F821


class ForecastModuleConfig(object):
    """
    Configuration class for forecast training module.
    
    Unlike diffusion models, forecasting does not require:
    - Preconditioner (no noise scaling)
    - Noise sampler (no noise injection)
    - Noise scheduler (no diffusion steps)
    
    Instead, it focuses on:
    - Loss configuration for direct prediction
    - Optional autoencoder for latent space
    - Optional dynamic weighting or masking
    """
    
    def __init__(self,
                 loss_metric: Union[str, Dict[str, Any]] = "mse",
                 tag: str = "forecast",
                 has_autoencoder_normalization: bool = False,
                 dynamic_loss_weight: Optional[int] = None,
                 loss_in_latent_space: bool = False,
                 freeze_autoencoder: bool = True,
                 extra_args: Optional[Dict[str, Any]] = None,
                 # Optional spatial weighting for region-specific losses
                 spatial_weight_map: Optional[Tensor] = None):
        """
        Parameters
        ----------
        loss_metric : str or dict, default="mse"
            Loss configuration. Can be:
            - str: "mse", "mae", "huber"
            - dict: {"mse": {}}, {"huber": {"delta": 1.0}}, etc.
        tag : str, default="forecast"
            Configuration tag for identification
        has_autoencoder_normalization : bool, default=False
            Whether to apply normalization before/after autoencoder
        dynamic_loss_weight : int, optional
            Dimension for dynamic loss weighting (experimental)
        loss_in_latent_space : bool, default=False
            If True and using autoencoder, compute loss in latent space (no decoding).
            If False, decode to pixel space before computing loss.
            Only applies when autoencoder is used.
        freeze_autoencoder : bool, default=True
            Whether to freeze autoencoder weights during training.
            Set to True if autoencoder is pre-trained (recommended).
            Set to False to fine-tune autoencoder jointly with model.
        extra_args : dict, optional
            Extra arguments for configuration
        spatial_weight_map : Tensor, optional
            Spatial weights for region-specific losses (e.g., focus on coastal regions)
        """
        self.loss_metric = loss_metric
        self.tag = tag
        self.has_autoencoder_normalization = has_autoencoder_normalization
        self.dynamic_loss_weight = dynamic_loss_weight
        self.loss_in_latent_space = loss_in_latent_space
        self.freeze_autoencoder = freeze_autoencoder
        self.spatial_weight_map = spatial_weight_map
        
        if extra_args is None:
            self.extra_args = {}
        else:
            self.extra_args = extra_args

    @classmethod
    def from_simple(cls,
                    loss_metric: str = "mse",
                    has_autoencoder_normalization: bool = False,
                    loss_in_latent_space: bool = False,
                    freeze_autoencoder: bool = True):
        """
        Create a simple forecast configuration.
        
        Parameters
        ----------
        loss_metric : str, default="mse"
            Loss function name
        has_autoencoder_normalization : bool, default=False
            Whether to normalize in latent space
        loss_in_latent_space : bool, default=False
            Compute loss in latent space (no pixel-space decoding)
        freeze_autoencoder : bool, default=True
            Freeze autoencoder weights (recommended for pre-trained AE)
            
        Returns
        -------
        ForecastModuleConfig
            Configuration instance
        """
        return cls(
            loss_metric=loss_metric,
            tag="forecast_simple",
            has_autoencoder_normalization=has_autoencoder_normalization,
            loss_in_latent_space=loss_in_latent_space,
            freeze_autoencoder=freeze_autoencoder
        )

    @classmethod
    def from_advanced(cls,
                      loss_metric: Union[str, Dict[str, Any]] = "huber",
                      has_autoencoder_normalization: bool = True,
                      dynamic_loss_weight: Optional[int] = 32,
                      loss_in_latent_space: bool = False,
                      freeze_autoencoder: bool = True,
                      spatial_weight_map: Optional[Tensor] = None):
        """
        Create an advanced forecast configuration with dynamic weighting.
        
        Parameters
        ----------
        loss_metric : str or dict, default="huber"
            Loss configuration
        has_autoencoder_normalization : bool, default=True
            Enable autoencoder normalization
        dynamic_loss_weight : int, optional
            Fourier feature dimension for dynamic weighting
        loss_in_latent_space : bool, default=False
            Compute loss in latent space (no pixel-space decoding)
        freeze_autoencoder : bool, default=True
            Freeze autoencoder weights (recommended for pre-trained AE)
        spatial_weight_map : Tensor, optional
            Spatial weighting (e.g., coastal emphasis)
            
        Returns
        -------
        ForecastModuleConfig
            Advanced configuration instance
        """
        return cls(
            loss_metric=loss_metric,
            tag="forecast_advanced",
            has_autoencoder_normalization=has_autoencoder_normalization,
            dynamic_loss_weight=dynamic_loss_weight,
            loss_in_latent_space=loss_in_latent_space,
            freeze_autoencoder=freeze_autoencoder,
            spatial_weight_map=spatial_weight_map
        )

    def export_description(self) -> Dict[str, Any]:
        """Export configuration for saving."""
        return {
            'tag': self.tag,
            'loss_metric': self.loss_metric,
            'has_autoencoder_normalization': self.has_autoencoder_normalization,
            'dynamic_loss_weight': self.dynamic_loss_weight,
            'loss_in_latent_space': self.loss_in_latent_space,
            'freeze_autoencoder': self.freeze_autoencoder,
            'extra_args': self.extra_args
        }

    @classmethod
    def from_description(cls, description: Dict[str, Any]) -> 'ForecastModuleConfig':
        """Load configuration from description."""
        return cls(
            loss_metric=description.get('loss_metric', 'mse'),
            tag=description.get('tag', 'forecast'),
            has_autoencoder_normalization=description.get('has_autoencoder_normalization', False),
            dynamic_loss_weight=description.get('dynamic_loss_weight', None),
            loss_in_latent_space=description.get('loss_in_latent_space', False),
            freeze_autoencoder=description.get('freeze_autoencoder', True),
            extra_args=description.get('extra_args', {})
        )


class ForecastModule(lightning.LightningModule):
    """
    Lightning module for deterministic ocean forecasting.
    
    Features:
    - Direct prediction (no diffusion process)
    - Optional latent space training with autoencoder
    - Multi-loss support
    - Conditional embeddings
    - Masking for incomplete regions
    - Comprehensive logging
    
    Parameters
    ----------
    model : torch.nn.Module
        Forecast network (e.g., OUNetGForecast)
    config : ForecastModuleConfig
        Training configuration
    conditional : bool, default=False
        Whether model expects conditional input (y parameter)
    masked : bool, default=False
        Whether data has masks for missing regions
    autoencoder : torch.nn.Module, optional
        Autoencoder for latent space training. If provided:
        - Input is encoded to latent space
        - Model operates in latent space
        - Output is decoded back to original space
    autoencoder_conditional : bool, default=False
        Whether autoencoder is conditional (encoder takes y)
    encode_y : bool, default=False
        Whether to encode the conditional y through autoencoder
    """
    
    def __init__(self,
                 model: torch.nn.Module,
                 config: ForecastModuleConfig,
                 conditional: bool = False,
                 masked: bool = False,
                 autoencoder: Optional[torch.nn.Module] = None,
                 autoencoder_conditional: bool = False,
                 encode_y: bool = False):
        super().__init__()
        self.model = model
        self.config = config
        self.conditional = conditional
        self.masked = masked
        self.autoencoder = autoencoder
        self.autoencoder_conditional = autoencoder_conditional
        self.encode_y = encode_y
        
        # Freeze autoencoder if configured and provided
        if self.autoencoder is not None and self.config.freeze_autoencoder:
            self.freeze_autoencoder()
        
        # Setup training components
        self.set_optimizer_and_scheduler()
        self.set_loss_metric()
        self.norm = 1.0
        
        # Metrics storage for logging
        self.train_losses = []
        self.val_losses = []

    def freeze_autoencoder(self):
        """
        Freeze autoencoder weights to prevent training updates.
        
        Use this when:
        - Autoencoder is pre-trained and should not be fine-tuned
        - You want to train only the forecast model
        - You want to save GPU memory (less gradients to track)
        
        This is automatically called during initialization if:
        - autoencoder is provided
        - config.freeze_autoencoder=True (default)
        """
        if self.autoencoder is None:
            return
        
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def unfreeze_autoencoder(self):
        """
        Unfreeze autoencoder weights to allow training updates.
        
        Use this when:
        - You want to fine-tune the autoencoder jointly with the model
        - You want to adapt autoencoder to ocean data
        - You have high-quality training data
        
        Warning: This increases memory usage and training time significantly.
        Only use if you have a good reason to fine-tune the autoencoder.
        """
        if self.autoencoder is None:
            return
        
        for param in self.autoencoder.parameters():
            param.requires_grad = True

    def export_description(self) -> Dict[str, Any]:
        """Export module configuration for saving."""
        return {
            'config_description': self.config.export_description(),
            'conditional': self.conditional,
            'masked': self.masked,
            'autoencoder': self.autoencoder is not None,
            'autoencoder_conditional': self.autoencoder_conditional,
            'encode_y': self.encode_y,
            'loss_in_latent_space': self.config.loss_in_latent_space,
            'freeze_autoencoder': self.config.freeze_autoencoder
        }

    def set_optimizer_and_scheduler(self,
                                   optimizer: Optional[torch.optim.Optimizer] = None,
                                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                                   scheduler_interval: str = "epoch"):
        """
        Set up optimizer and learning rate scheduler.
        
        Parameters
        ----------
        optimizer : Optimizer, optional
            If None, use AdamW with default parameters
        scheduler : LRScheduler, optional
            If None, use identity scheduler (no scheduling)
        scheduler_interval : str, default="epoch"
            "epoch" or "step" - when to apply scheduler
        """
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=1e-3,
                betas=(0.9, 0.999),
                weight_decay=1e-4
            )
        
        if scheduler is not None:
            self.lr_scheduler = scheduler
        else:
            # Identity scheduler - no changes to learning rate
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: 1.0
            )
        
        self.lr_scheduler_interval = scheduler_interval

    def set_loss_metric(self):
        """
        Initialize loss function based on configuration.
        
        Supports:
        - str: "mse", "mae", "huber"
        - dict: {"mse": {}}, {"huber": {"delta": 1.0}}, etc.
        """
        loss_config = self.config.loss_metric
        
        if isinstance(loss_config, str):
            self._set_single_loss_string(loss_config)
        elif isinstance(loss_config, dict):
            self._set_single_loss_dict(loss_config)
        else:
            raise ValueError(f"loss_metric must be string or dict, got {type(loss_config)}")

    def _set_single_loss_string(self, loss_name: str):
        """Set loss from string name."""
        if loss_name == "mse":
            self.loss_metric = torch.nn.MSELoss(reduction="none")
        elif loss_name == "mae":
            self.loss_metric = torch.nn.L1Loss(reduction="none")
        elif loss_name == "huber":
            self.loss_metric = torch.nn.HuberLoss(reduction="none", delta=1.0)
        elif loss_name == "smooth_l1":
            self.loss_metric = torch.nn.SmoothL1Loss(reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {loss_name}")

    def _set_single_loss_dict(self, loss_config: Dict[str, Any]):
        """Set loss from dictionary configuration."""
        loss_name = list(loss_config.keys())[0]
        loss_params = loss_config[loss_name]
        
        if loss_name == "mse":
            self.loss_metric = torch.nn.MSELoss(reduction="none")
        elif loss_name == "mae":
            self.loss_metric = torch.nn.L1Loss(reduction="none")
        elif loss_name == "huber":
            delta = loss_params.get('delta', 1.0)
            self.loss_metric = torch.nn.HuberLoss(reduction="none", delta=delta)
        elif loss_name == "smooth_l1":
            beta = loss_params.get('beta', 1.0)
            self.loss_metric = torch.nn.SmoothL1Loss(reduction="none", beta=beta)
        else:
            raise ValueError(f"Unknown loss type: {loss_name}")

    def encode(self, x: Tensor, y: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Encode input to latent space using autoencoder (if available).
        
        Parameters
        ----------
        x : Tensor
            Input data [B, C, H, W]
        y : Tensor, optional
            Conditional data (used if autoencoder_conditional=True)
            
        Returns
        -------
        x_latent : Tensor
            Encoded data in latent space
        y : Tensor, optional (if encode_y=True)
            Encoded conditional data
        """
        if self.latent_model:
            if self.autoencoder_conditional:
                if self.encode_y:
                    x, y = self.autoencoder.encode(x, y)
                else:
                    x = self.autoencoder.encode(x, y)
            else:
                x = self.autoencoder.encode(x)
        
        # Apply normalization if configured
        if self.config.has_autoencoder_normalization:
            x = x / self.norm
        
        if self.encode_y:
            return x, y
        else:
            return x

    def decode(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        Decode from latent space back to original space.
        
        Parameters
        ----------
        x : Tensor
            Data in latent space
        y : Tensor, optional
            Conditional data (used if autoencoder_conditional=True)
            
        Returns
        -------
        x_pixel : Tensor
            Data in original space
        """
        # Reverse normalization
        if self.config.has_autoencoder_normalization:
            x = x * self.norm
        
        # Decode from latent space
        if self.latent_model:
            if self.autoencoder_conditional:
                x = self.autoencoder.decode(x, y)
            else:
                x = self.autoencoder.decode(x)
        
        return x

    def loss_fn(self,
                pred: Tensor,
                target: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        """
        Compute loss between prediction and target.
        
        Parameters
        ----------
        pred : Tensor
            Model prediction [B, C, H, W]
        target : Tensor
            Target/ground truth [B, C, H, W]
        mask : Tensor, optional
            Binary mask for regions to include in loss (1=include, 0=exclude)
            
        Returns
        -------
        loss : Tensor
            Scalar loss value
        """
        # Compute element-wise loss
        loss = self.loss_metric(pred, target)  # [B, C, H, W]
        
        # Apply mask if provided (1=valid, 0=invalid)
        if mask is not None:
            # Ensure mask has same shape as loss
            if mask.dim() < loss.dim():
                # Broadcast mask to match loss dimensions
                while mask.dim() < loss.dim():
                    mask = mask.unsqueeze(1)
            mask_expanded = mask.expand_as(loss)
            loss = loss * mask_expanded
        
        # Apply spatial weights if configured
        if self.config.spatial_weight_map is not None:
            weight_map = self.config.spatial_weight_map.to(loss.device)
            if weight_map.dim() < loss.dim():
                while weight_map.dim() < loss.dim():
                    weight_map = weight_map.unsqueeze(0).unsqueeze(0)
            weight_map = weight_map.expand_as(loss)
            loss = loss * weight_map
        
        # Average over batch and spatial dimensions
        return loss.mean()

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through the model.
        
        Parameters
        ----------
        x : Tensor
            Input [B, C, H, W]
        y : Tensor, optional
            Conditional data (if conditional=True)
            
        Returns
        -------
        output : Tensor
            Prediction [B, C, H, W]
        """
        if self.conditional and y is not None:
            return self.model(y['y'], y=y)
        else:
            if x is None:
                raise ValueError("x required for non-conditional")
            return self.model(x, y=y)

    def training_step(self, batch, batch_idx: int) -> Tensor:
        """
        Training step for one batch.
        
        Batch format depends on configuration:
        - (x, y) if conditional and not masked
        - (x, y, mask) if conditional and masked
        - (x, mask) if not conditional and masked
        - (x,) if not conditional and not masked
        
        Loss computation:
        - If loss_in_latent_space=True and autoencoder is present:
          Compute loss in latent space (no decoding)
        - If loss_in_latent_space=False and autoencoder is present:
          Decode to pixel space before computing loss
        - If no autoencoder:
          Compute loss in model output space
        """
        x, y, mask = self.select_batch(batch)
        
        # Encode to latent space if autoencoder is available
        if self.encode_y:
            x_encoded, y = self.encode(x, y)
        else:
            x_encoded = self.encode(x, y)
        
        # Forward pass in latent/model space
        if self.conditional and y is not None:
            pred_latent = self.model(y['y'], y=y)
        # TODO: Recode this to remove the x from the diffusion case 
        else:
            raise ValueError("This model is always conditional, must provide y")
        
        # Compute loss
        if self.config.loss_in_latent_space and self.latent_model:
            # Loss in latent space (no decoding)
            loss = self.loss_fn(pred_latent, x_encoded, mask)
        else:
            # Decode to pixel space and compute loss there
            if self.latent_model:
                target = self.decode(x_encoded, y)
                pred = self.decode(pred_latent, y)
            else:
                target = x_encoded
                pred = pred_latent
            
            loss = self.loss_fn(pred, target, mask)
        
        # Logging
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.train_losses.append(loss.detach().item())
        
        return loss

    def validation_step(self, batch, batch_idx: int) -> Tensor:
        """
        Validation step for one batch.
        
        Loss computation strategy matches training_step:
        - If loss_in_latent_space=True and autoencoder is present:
          Compute loss in latent space (no decoding)
        - If loss_in_latent_space=False and autoencoder is present:
          Decode to pixel space before computing loss
        - If no autoencoder:
          Compute loss in model output space
        """
        x, y, mask = self.select_batch(batch)
        
        # Encode to latent space
        if self.encode_y:
            x_encoded, y = self.encode(x, y)
        else:
            x_encoded = self.encode(x, y)
        
        # Forward pass
        pred_latent = self.forward(x_encoded, y)
        
        # Compute loss
        if self.config.loss_in_latent_space and self.latent_model:
            # Loss in latent space (no decoding)
            loss = self.loss_fn(pred_latent, x_encoded, mask)
        else:
            # Decode to pixel space and compute loss there
            if self.latent_model:
                target = self.decode(x_encoded, y)
                pred = self.decode(pred_latent, y)
            else:
                target = x_encoded
                pred = pred_latent
            
            loss = self.loss_fn(pred, target, mask)
        
        # Logging
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('valid_loss', loss, prog_bar=True, sync_dist=True)  # Backward compatibility
        self.val_losses.append(loss.detach().item())
        
        return loss

    def configure_optimizers(self) -> Union[torch.optim.Optimizer,
                                           Tuple[List[torch.optim.Optimizer],
                                                 List[Dict[str, Any]]]]:
        """
        Configure optimizers and schedulers for Lightning.
        
        Returns
        -------
        optimizer or (optimizers, schedulers)
            Optimizer configuration for Lightning
        """
        if self.lr_scheduler is not None:
            lr_scheduler_config = {
                "scheduler": self.lr_scheduler,
                "interval": self.lr_scheduler_interval
            }
            return [self.optimizer], [lr_scheduler_config]
        else:
            return self.optimizer

    def select_batch(self, batch) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Unpack batch based on configuration.
        
        Returns
        -------
        x : Tensor
            Input data
        y : Tensor or None
            Conditional data (or None if not conditional)
        mask : Tensor or None
            Mask for missing regions (or None if not masked)
        """
        if self.conditional and self.masked:
            x, y, mask = batch
        elif (not self.conditional) and self.masked:
            x, mask = batch
            y = None
        elif self.conditional and (not self.masked):
            x, y = batch
            mask = None
        else:
            x = batch
            y = None
            mask = None
        
        return x, y, mask

    @property
    def latent_model(self) -> bool:
        """Whether this module uses an autoencoder (latent space training)."""
        return self.autoencoder is not None

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Prediction step for inference.
        
        Returns:
        - If loss_in_latent_space=True and autoencoder present:
          Returns latent space prediction (no decoding)
        - If loss_in_latent_space=False and autoencoder present:
          Returns pixel space prediction (after decoding)
        - If no autoencoder:
          Returns model output
          
        If encode_y=True, returns tuple (predictions, encoded_y)
        """
        x, y, _ = self.select_batch(batch)
        
        # Encode
        if self.latent_model:
            if self.encode_y:
                x_encoded, y = self.encode(x, y)
            else:
                x_encoded = self.encode(x, y)
        else:
            x_encoded = x
        
        # Predict in latent space
        pred_latent = self.forward(x_encoded, y)
        
        # Return prediction based on configuration
        if self.config.loss_in_latent_space and self.latent_model:
            # Return latent space prediction (no decoding)
            if self.encode_y:
                return pred_latent, y
            else:
                return pred_latent
        else:
            # Decode to original space
            if self.latent_model:
                pred = self.decode(pred_latent, y)
            else:
                pred = pred_latent
            
            if self.encode_y:
                return pred, y
            else:
                return pred

    def sample(self, y:  dict[str, Tensor], return_latent: bool = False) -> Tensor:
        """
        Sample from the model.

        Parameters
        ----------
        y : dict[str, Tensor]
            Conditional data (if conditional=True)
        return_latent : bool, default=False
            Whether to return the latent space prediction
        Returns
        -------
        pred : Tensor
            Prediction [B, C, H, W]
        """

        # Encode
        #check dimension of y['y']
        if y['y'].dim() == 3:
            y['y'] = y['y'].unsqueeze(0)
        #check dimension of y['bat']
        if y['bat'].dim() == 3:
            y['bat'] = y['bat'].unsqueeze(0)
        
        #check dimension of y['latlon']
        if y['latlon'].dim() == 1:
            y['latlon'] = y['latlon'].reshape(1, 2)

        

        x = y['y'][:,-3:]
        if self.latent_model:
            if self.encode_y:
                x_encoded, y = self.encode(x, y)
            else:
                x_encoded = self.encode(x, y)
        else:
            x_encoded = x
        
        # Predict in latent space
        pred_latent = self.forward(y['y'], y)
        
        # Return prediction based on configuration
        if return_latent:
            # Return latent space prediction (no decoding)
            if self.encode_y:
                return pred_latent, y
            else:
                return pred_latent
        else:
            # Decode to original space
            if self.latent_model:
                pred = self.decode(pred_latent, y)
            else:
                pred = pred_latent
            
            return pred


class DynamicLossWeight(torch.nn.Module):
    """
    Dynamic loss weighting using Fourier features.
    
    Creates learnable frequency-dependent weights for loss scaling.
    Useful for emphasizing different spectral components.
    
    Parameters
    ----------
    nhidden : int
        Number of hidden Fourier features
    scale : float, default=1.0
        Initialization scale
    """
    
    def __init__(self, nhidden: int, scale: float = 1.0):
        super().__init__()
        self.nhidden = nhidden
        
        # Fourier feature weights
        self.register_buffer(
            "fourier_weights",
            torch.randn(nhidden) * scale
        )
        self.register_buffer(
            "fourier_bias",
            torch.rand(nhidden) * scale
        )
        
        # Linear layer to combine features
        self.linear = torch.nn.Linear(nhidden, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute dynamic weight for given input.
        
        Parameters
        ----------
        x : Tensor of shape [batch]
            Input value
            
        Returns
        -------
        weight : Tensor of shape [batch]
            Dynamic weight for each sample
        """
        x = x.unsqueeze(1)  # [batch, 1]
        h = x * self.fourier_weights + self.fourier_bias  # [batch, nhidden]
        h = torch.cos(h)  # [batch, nhidden]
        h = self.linear(h)  # [batch, 1]
        return h.squeeze(1)  # [batch]