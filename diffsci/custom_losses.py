import torch
import torch.nn as nn
from typing import Tuple, Union, Optional, List, Dict, Any

class GaussianWeightedMSELoss(nn.Module):
    """
    A general-purpose Mean Squared Error (MSE) loss that applies a Gaussian
    weight mask to N-dimensional spatial data. This allows the model to focus
    more on the center of the spatial dimensions (e.g., D, H, W).

    """
    def __init__(self, shape: Tuple[int, ...], focus_radius: float, device: str = 'cpu'):
        """
        Initializes the GaussianWeightedMSELoss module.

        Args:
            shape (Tuple[int, ...]): A tuple representing the spatial dimensions
                of the input tensors (e.g., (H, W) for 2D, or (D, H, W) for 3D).
            focus_radius (float): A parameter controlling the standard deviation
                of the Gaussian. A smaller value creates a sharper focus.
            device (str): The device to store the weight mask on ('cpu' or 'cuda').
        """
        super(GaussianWeightedMSELoss, self).__init__()
        self.shape = shape
        self.focus_radius = focus_radius
        
        # Create the N-dimensional Gaussian weight mask and register it as a buffer.
        weight_mask = self._create_gaussian_window(shape, focus_radius)
        self.register_buffer('weight_mask', weight_mask.to(device))

    def _create_gaussian_window(self, shape: Tuple[int, ...], radius: float) -> torch.Tensor:
        """
        Creates an N-dimensional Gaussian distribution to be used as a weight mask.
        
        Args:
            shape (Tuple[int, ...]): The spatial dimensions of the window.
            radius (float): The radius which controls the sigma of the Gaussian.

        Returns:
            torch.Tensor: A tensor of shape [1, 1, *shape] containing the
                          Gaussian weights, with values between 0 and 1.
        """
        # The standard deviation (sigma) of the Gaussian.
        # CORRECTED: Now uses the 'radius' parameter passed to the method.
        sigma = radius + 1e-8

        # Create a 1D coordinate tensor for each spatial dimension
        coords = [torch.linspace(-1, 1, s) for s in shape]
        
        # Create an N-dimensional coordinate grid
        # The 'ij' indexing ensures the grid axes match the tensor dimensions order
        grids = torch.meshgrid(*coords, indexing='ij')
        
        # Calculate the squared Euclidean distance from the center (0,0,...,0)
        distance_squared = torch.zeros_like(grids[0])
        for grid in grids:
            distance_squared += grid**2
        
        # Calculate the N-dimensional Gaussian
        gaussian_weights = torch.exp(-distance_squared / (2 * sigma**2))
        
        # Reshape to [1, 1, *shape] to be broadcastable with input tensors [B, C, *shape]
        # e.g., for 3D data, this will be [1, 1, D, H, W]
        final_shape = (1, 1) + shape
        return gaussian_weights.view(final_shape)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the weighted squared error without reduction. This makes it
        compatible with frameworks like the KarrasModule that apply other
        weights before the final reduction.

        Args:
            input (torch.Tensor): The predicted tensor. Shape: [B, C, *shape].
            target (torch.Tensor): The ground truth tensor. Shape: [B, C, *shape].

        Returns:
            torch.Tensor: A tensor of shape [B, C, *shape] containing the
                          weighted squared errors.
        """
        # Ensure the weight mask is on the same device as the input tensor
        if self.weight_mask.device != input.device:
            self.weight_mask = self.weight_mask.to(input.device)

        # 1. Calculate the standard element-wise squared error
        squared_error = (input - target) ** 2
        
        # 2. Apply the Gaussian weights by element-wise multiplication.
        # The weight_mask [1, 1, *shape] will be broadcast across the
        # batch and channel dimensions of the squared_error [B, C, *shape].
        weighted_squared_error = squared_error * self.weight_mask
        
        # 3. Return the weighted error tensor without reduction
        return weighted_squared_error

class MultiThresholdSmoothIndicatorLoss(torch.nn.Module):
    """
    Smooth approximation of indicator function loss for multiple threshold high current detection.
    
    This loss focuses on whether the model correctly predicts high currents above
    multiple thresholds, rather than exact values. Uses smooth approximations to
    maintain differentiability and handles False Positives appropriately.
    """
    
    def __init__(self, 
                 thresholds: Union[List[float], float],
                 temperature: float = 10.0,
                 loss_type: str = 'sigmoid',
                 focus_weights: Optional[Union[List[float], float]] = None,
                 background_weights: Optional[Union[List[float], float]] = None,
                 fp_penalty: float = 1.0,
                 se_weight: float = 0.1,
                 aggregation: str = 'mean'):
        """
        Args:
            thresholds: Single threshold or list of current magnitude thresholds
            temperature: Controls smoothness (higher = sharper transition)
            loss_type: Type of smooth approximation ('sigmoid', 'tanh', 'gumbel')
            focus_weights: Weight for high current regions (above each threshold)
            background_weights: Weight for low current regions (below each threshold)
            fp_penalty: Penalty weight for False Positives (pred high, target low)
            se_weight: Weight for squared error component
            aggregation: How to combine losses across thresholds ('mean', 'sum', 'max')
        """
        super().__init__()
        
        # Convert single threshold to list
        if isinstance(thresholds, (int, float)):
            self.thresholds = [float(thresholds)]
        else:
            self.thresholds = list(thresholds)
        
        self.n_thresholds = len(self.thresholds)
        self.temperature = temperature
        self.loss_type = loss_type
        self.fp_penalty = fp_penalty
        self.se_weight = se_weight
        self.aggregation = aggregation
        
        # Handle weights for each threshold
        if focus_weights is None:
            self.focus_weights = [2.0] * self.n_thresholds
        elif isinstance(focus_weights, (int, float)):
            self.focus_weights = [float(focus_weights)] * self.n_thresholds
        else:
            assert len(focus_weights) == self.n_thresholds, \
                f"focus_weights length {len(focus_weights)} != thresholds length {self.n_thresholds}"
            self.focus_weights = list(focus_weights)
        
        if background_weights is None:
            self.background_weights = [0.1] * self.n_thresholds
        elif isinstance(background_weights, (int, float)):
            self.background_weights = [float(background_weights)] * self.n_thresholds
        else:
            assert len(background_weights) == self.n_thresholds, \
                f"background_weights length {len(background_weights)} != thresholds length {self.n_thresholds}"
            self.background_weights = list(background_weights)
    
    def smooth_indicator(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Smooth approximation of indicator function I(x > threshold)"""
        if self.loss_type == 'sigmoid':
            return torch.sigmoid(self.temperature * (x - threshold))
        elif self.loss_type == 'tanh':
            return 0.5 * (1 + torch.tanh(self.temperature * (x - threshold)))
        elif self.loss_type == 'gumbel':
            # Gumbel softmax approximation
            logits = torch.stack([torch.zeros_like(x), 
                                 self.temperature * (x - threshold)], dim=-1)
            return torch.nn.functional.softmax(logits, dim=-1)[..., 1]
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
    
    def compute_threshold_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                              threshold: float, focus_weight: float, 
                              background_weight: float,
                              mask: Optional[torch.Tensor] = None) -> dict:
        """Compute loss for a single threshold with proper BCE and squared error"""
        
        # Smooth indicators
        target_indicator = self.smooth_indicator(target, threshold)
        pred_indicator = self.smooth_indicator(pred, threshold)
        
        eps = 1e-8
        
        # Standard Binary Cross-Entropy Loss
        bce_loss = -(target_indicator * torch.log(pred_indicator + eps) + 
                    (1 - target_indicator) * torch.log(1 - pred_indicator + eps))
        
        # Apply False Positive penalty: extra penalty when target=0 but pred=1
        fp_penalty_term = (1 - target_indicator) * pred_indicator * (self.fp_penalty - 1.0)
        
        # Combined BCE + FP penalty
        indicator_loss = bce_loss + fp_penalty_term
        
        # Squared Error Loss (for value accuracy)
        squared_error = (pred - target) ** 2
        
        # Apply spatial weighting
        weighted_indicator_loss = (focus_weight * indicator_loss * target_indicator + 
                                  background_weight * indicator_loss * (1 - target_indicator))
        
        # Weight squared error by target intensity (focus more on high current regions)
        weighted_squared_error = squared_error * (1.0 + target_indicator)
        
        # Apply mask if provided
        if mask is not None:
            valid_mask = (~mask).float()
            weighted_indicator_loss = weighted_indicator_loss * valid_mask
            weighted_squared_error = weighted_squared_error * valid_mask
            
            # Compute detailed statistics for analysis
            # TP: target high (>0.5), pred high (>0.5)
            tp_mask = (target_indicator > 0.5) & (pred_indicator > 0.5) & valid_mask.bool()
            # FP: target low (<=0.5), pred high (>0.5)  
            fp_mask = (target_indicator <= 0.5) & (pred_indicator > 0.5) & valid_mask.bool()
            # FN: target high (>0.5), pred low (<=0.5)
            fn_mask = (target_indicator > 0.5) & (pred_indicator <= 0.5) & valid_mask.bool()
            # TN: target low (<=0.5), pred low (<=0.5)
            tn_mask = (target_indicator <= 0.5) & (pred_indicator <= 0.5) & valid_mask.bool()
            
            stats = {
                'indicator_loss': weighted_indicator_loss.sum() / valid_mask.sum().clamp(min=1),
                'squared_error': weighted_squared_error.sum() / valid_mask.sum().clamp(min=1),
                'bce_loss': (bce_loss * valid_mask).sum() / valid_mask.sum().clamp(min=1),
                'fp_penalty': (fp_penalty_term * valid_mask).sum() / valid_mask.sum().clamp(min=1),
                'tp_count': tp_mask.sum().float(),
                'fp_count': fp_mask.sum().float(),
                'fn_count': fn_mask.sum().float(),
                'tn_count': tn_mask.sum().float(),
                'tp_avg_pred': pred_indicator[tp_mask].mean() if tp_mask.any() else torch.tensor(0.0),
                'fp_avg_pred': pred_indicator[fp_mask].mean() if fp_mask.any() else torch.tensor(0.0),
                'fn_avg_pred': pred_indicator[fn_mask].mean() if fn_mask.any() else torch.tensor(0.0),
                'tn_avg_pred': pred_indicator[tn_mask].mean() if tn_mask.any() else torch.tensor(0.0),
                'n_high_target': (target_indicator > 0.5).sum().float(),
                'n_low_target': (target_indicator <= 0.5).sum().float()
            }
        else:
            # Without mask
            tp_mask = (target_indicator > 0.5) & (pred_indicator > 0.5)
            fp_mask = (target_indicator <= 0.5) & (pred_indicator > 0.5)
            fn_mask = (target_indicator > 0.5) & (pred_indicator <= 0.5)
            tn_mask = (target_indicator <= 0.5) & (pred_indicator <= 0.5)
            
            stats = {
                'indicator_loss': weighted_indicator_loss.mean(),
                'squared_error': weighted_squared_error.mean(),
                'bce_loss': bce_loss.mean(),
                'fp_penalty': fp_penalty_term.mean(),
                'tp_count': tp_mask.sum().float(),
                'fp_count': fp_mask.sum().float(),
                'fn_count': fn_mask.sum().float(),
                'tn_count': tn_mask.sum().float(),
                'tp_avg_pred': pred_indicator[tp_mask].mean() if tp_mask.any() else torch.tensor(0.0),
                'fp_avg_pred': pred_indicator[fp_mask].mean() if fp_mask.any() else torch.tensor(0.0),
                'fn_avg_pred': pred_indicator[fn_mask].mean() if fn_mask.any() else torch.tensor(0.0),
                'tn_avg_pred': pred_indicator[tn_mask].mean() if tn_mask.any() else torch.tensor(0.0),
                'n_high_target': (target_indicator > 0.5).sum().float(),
                'n_low_target': (target_indicator <= 0.5).sum().float()
            }
        
        return stats
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: Predicted current magnitudes [batch, ...]
            target: Target current magnitudes [batch, ...]
            mask: Optional mask for valid regions [batch, ...]
        
        Returns:
            Single tensor loss value compatible with your training setup
        """
        
        all_losses = []
        
        for i, threshold in enumerate(self.thresholds):
            stats = self.compute_threshold_loss(
                pred, target, threshold, 
                self.focus_weights[i], self.background_weights[i], mask
            )
            
            # Combine indicator loss and squared error
            combined_loss = stats['indicator_loss'] + self.se_weight * stats['squared_error']
            all_losses.append(combined_loss)
        
        # Aggregate losses across thresholds
        if self.aggregation == 'mean':
            total_loss = torch.stack(all_losses).mean()
        elif self.aggregation == 'sum':
            total_loss = torch.stack(all_losses).sum()
        elif self.aggregation == 'max':
            total_loss = torch.stack(all_losses).max()
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        return total_loss
    
    def forward_with_stats(self, pred: torch.Tensor, target: torch.Tensor, 
                          mask: Optional[torch.Tensor] = None) -> dict:
        """
        Alternative forward method that returns detailed statistics.
        Use this for analysis/debugging, but use forward() for training.
        """
        
        all_losses = []
        threshold_stats = {}
        
        for i, threshold in enumerate(self.thresholds):
            stats = self.compute_threshold_loss(
                pred, target, threshold, 
                self.focus_weights[i], self.background_weights[i], mask
            )
            
            # Combine indicator loss and squared error
            combined_loss = stats['indicator_loss'] + self.se_weight * stats['squared_error']
            all_losses.append(combined_loss)
            threshold_stats[f'threshold_{threshold}'] = stats
        
        # Aggregate losses across thresholds
        if self.aggregation == 'mean':
            total_loss = torch.stack(all_losses).mean()
        elif self.aggregation == 'sum':
            total_loss = torch.stack(all_losses).sum()
        elif self.aggregation == 'max':
            total_loss = torch.stack(all_losses).max()
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        result = {
            'total_loss': total_loss,
            'individual_losses': all_losses,
            'threshold_stats': threshold_stats,
            'thresholds': self.thresholds
        }
        
        return result

class MultiSpaceLoss:
    """
    Handles multiple losses that can be applied in different spaces (latent/pixel).
    Automatically detects if losses handle masks internally or need external mask handling.
    
    Configuration format:
    {
        "losses": [
            {
                "name": "mse_latent",
                "type": "mse", 
                "space": "latent",
                "weight": 1.0,
                "use_mask": False
            },
            {
                "name": "smoothed_indicator_pixel",
                "type": "smoothed_indicator",  # Handles mask internally
                "space": "pixel", 
                "weight": 0.5,
                "use_mask": True,
                "params": {
                    "thresholds": [0.5, 0.7, 1.0],
                    "fp_penalty": 1.5
                }
            }
        ]
    }
    """
    
    def __init__(self, loss_config: Dict[str, Any], autoencoder=None):
        """
        Args:
            loss_config: Dictionary containing loss configuration
            autoencoder: Autoencoder for latent<->pixel conversion
        """
        self.autoencoder = autoencoder
        self.losses = []
        
        for loss_spec in loss_config["losses"]:
            loss_instance = self._create_loss_instance(loss_spec)
            self.losses.append({
                "name": loss_spec["name"],
                "loss_fn": loss_instance,
                "space": loss_spec["space"],
                "weight": loss_spec.get("weight", 1.0),
                "use_mask": loss_spec.get("use_mask", True),
                "handles_mask_internally": self._check_if_handles_mask_internally(loss_spec["type"])
            })
    
    def _check_if_handles_mask_internally(self, loss_type: str) -> bool:
        """
        Check if a loss type handles masks internally and returns scalar loss.
        
        Returns:
            True if loss handles mask internally (returns scalar)
            False if loss needs external mask handling (returns tensor)
        """
        internal_mask_losses = {
            "smoothed_indicator",
            "multi_threshold_focal", 
            "multi_threshold_combined"
            # Add other custom losses that handle masks internally
        }
        
        return loss_type in internal_mask_losses
    
    def _create_loss_instance(self, loss_spec: Dict[str, Any]):
        """Create a loss function instance from specification"""
        loss_type = loss_spec["type"]
        params = loss_spec.get("params", {})
        
        if loss_type == "mse":
            return torch.nn.MSELoss(reduction="none")
        elif loss_type == "huber":
            delta = params.get("delta", 1.0)
            return torch.nn.HuberLoss(reduction="none", delta=delta)
        elif loss_type == "smoothed_indicator":
            from your_loss_module import MultiThresholdSmoothIndicatorLoss
            return MultiThresholdSmoothIndicatorLoss(**params)
        elif loss_type == "multi_threshold_focal":
            from your_loss_module import MultiThresholdFocalIndicatorLoss
            return MultiThresholdFocalIndicatorLoss(**params)
        elif loss_type == "multi_threshold_combined":
            from your_loss_module import MultiThresholdCombinedLoss
            return MultiThresholdCombinedLoss(**params)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def compute_loss(self, 
                    denoiser_latent: torch.Tensor,
                    target_latent: torch.Tensor, 
                    target_pixel: Optional[torch.Tensor] = None,
                    mask_latent: Optional[torch.Tensor] = None,
                    mask_pixel: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute all configured losses with smart mask handling.
        
        Args:
            denoiser_latent: Denoiser output in latent space
            target_latent: Target in latent space  
            target_pixel: Target in pixel space (optional, computed if needed)
            mask_latent: Mask in latent space (can be None)
            mask_pixel: Mask in pixel space (can be None)
            
        Returns:
            Dictionary with individual and total losses
        """
        
        # Decode to pixel space if any loss needs it
        denoiser_pixel = None
        if any(loss["space"] == "pixel" for loss in self.losses):
            if self.autoencoder is None:
                raise ValueError("Autoencoder required for pixel space losses")
            denoiser_pixel = self.autoencoder.decode(denoiser_latent)
            
            if target_pixel is None:
                target_pixel = self.autoencoder.decode(target_latent)
        
        loss_values = {}
        total_loss = 0.0
        tensor_losses = []  # For losses that return tensors (need external mask handling)
        scalar_losses = []  # For losses that return scalars (handle masks internally)
        
        for loss_config in self.losses:
            name = loss_config["name"]
            loss_fn = loss_config["loss_fn"]
            space = loss_config["space"]
            weight = loss_config["weight"]
            use_mask = loss_config["use_mask"]
            handles_mask_internally = loss_config["handles_mask_internally"]
            
            # Select appropriate tensors based on space
            if space == "latent":
                pred = denoiser_latent
                target = target_latent
                mask = mask_latent if use_mask else None
            elif space == "pixel":
                pred = denoiser_pixel
                target = target_pixel
                mask = mask_pixel if use_mask else None
            else:
                raise ValueError(f"Unknown space: {space}")
            
            # Compute loss based on whether it handles masks internally
            if handles_mask_internally:
                # Loss handles mask internally (e.g., smoothed_indicator)
                try:
                    loss_value = loss_fn(pred, target, mask)
                except TypeError:
                    # Fallback if mask parameter not supported
                    loss_value = loss_fn(pred, target)
                
                # Should be scalar
                individual_loss = loss_value
                scalar_losses.append((weight, individual_loss))
                
            else:
                # Loss needs external mask handling (e.g., MSE)
                loss_value = loss_fn(pred, target)  # Don't pass mask
                
                # Apply mask externally if needed
                if mask is not None and use_mask:
                    mask_expanded = mask.expand_as(loss_value)
                    masked_loss = loss_value * (1 - mask_expanded)
                    # Reduce to scalar
                    individual_loss = masked_loss.sum() / (1 - mask_expanded).sum().clamp(min=1)
                else:
                    individual_loss = loss_value.mean()
                
                tensor_losses.append((weight, individual_loss))
            
            loss_values[name] = individual_loss
        
        # Combine all losses
        for weight, loss_val in scalar_losses + tensor_losses:
            total_loss += weight * loss_val
        
        loss_values["total"] = total_loss
        return loss_values