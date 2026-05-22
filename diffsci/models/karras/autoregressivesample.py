"""
FINAL CORRECTED VERSION: Encode y ONCE, never again

The issue: _encode_y_once() calls encode() which samples from VAE
Each call produces different conditioning!

Solution: In the CALLING CODE (not in autoregressive_sample),
encode y ONCE before passing to the function.

Two options:
1. Modify the function to accept PRE-ENCODED y
2. Or ensure the function only encodes once and never again
"""

import torch
from typing import Callable, Optional, Dict, List


class LatentSpaceAutoregressive:
    """
    Autoregressive sampling - CORRECTED for VAE randomness.
    
    KEY: y must be encoded ONCE before calling autoregressive_sample()
    NOT inside the function multiple times!
    """

    def autoregressive_sample(
        self,
        nsamples: int,
        latent_shape: List[int],
        nsteps_forecast: int,
        cond_time: int,
        nsteps_diffusion: int = 50,
        y: Optional[Dict[str, torch.Tensor]] = None,
        y_already_encoded: bool = False,  # ← NEW PARAMETER!
        guidance: float = 1.0,
        maximum_batch_size: Optional[int] = None,
        return_intermediate: bool = False,
        return_in_latent: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Autoregressive forecast.
        
        CRITICAL: Either:
        1. Pass y in PIXEL space and let us encode it ONCE
        2. Pass y already ENCODED and set y_already_encoded=True
        
        Args:
            nsamples: Batch size
            latent_shape: Shape in latent space
            nsteps_forecast: Number of autoregressive steps
            cond_time: Length of conditioning history
            nsteps_diffusion: Number of diffusion steps per forecast
            y: Conditioning dict (PIXEL space OR already encoded)
            y_already_encoded: If True, y is already encoded (skip encoding)
            guidance: Classifier-free guidance scale
            maximum_batch_size: For memory-efficient processing
            return_intermediate: If True, return all intermediate latent
            return_in_latent: If True, return forecasts in latent space
        
        Returns:
            Dict with forecasts
        """
        with torch.inference_mode():
            if maximum_batch_size is not None:
                return self._autoregressive_sample_batched(
                    nsamples,
                    latent_shape,
                    nsteps_forecast,
                    cond_time,
                    nsteps_diffusion,
                    y,
                    y_already_encoded,
                    guidance,
                    maximum_batch_size,
                    return_intermediate,
                    return_in_latent,
                )

            if y is None:
                y = {}
            
            y = dict(y)
            
            # Get pixel space dimensions
            if 'y' in y:
                y_shape = y['y'].shape
                pixel_h, pixel_w = y_shape[-2:]
                total_channels = y_shape[0]
                channels_per_step = total_channels // cond_time
            else:
                raise ValueError("y['y'] must be provided")
            
            # ========== CRITICAL FIX ==========
            # Only encode if not already encoded
            if not y_already_encoded:
                print("[autoregressive_sample] Encoding y ONCE...")
                y = self._encode_y_once(y)
            else:
                print("[autoregressive_sample] Using pre-encoded y (no encoding)")
            
            # After this point, y['y'] is in latent space
            # and we NEVER encode it again
            # ================================
            
            latent_c = latent_shape[0]
            latent_h = latent_shape[1]
            latent_w = latent_shape[2]
            
            max_buffer_size = max(cond_time, nsteps_forecast + 1)
            predictions_buffer_latent = torch.zeros(
                (max_buffer_size, nsamples, latent_c, latent_h, latent_w),
                dtype=torch.float32,
                device=self.device
            )
            
            forecasts_latent = []
            
            # Initial sample
            print(f"[autoregressive_sample] Sampling initial prediction...")
            x0_latent = self.sample(
                nsamples=nsamples,
                shape=latent_shape,
                y=y,
                guidance=guidance,
                nsteps=nsteps_diffusion,
                record_history=False,
                is_latent_shape=True,
                return_in_latent_space=True,
            )
            
            predictions_buffer_latent[0] = x0_latent
            forecasts_latent.append(x0_latent)
            
            # Autoregressive loop
            for step in range(nsteps_forecast-1):
                current_pred_count = step + 1
                
                # Update y['y'] with sliding window of latent predictions
                if current_pred_count >= cond_time:
                    start_idx = current_pred_count - cond_time
                    end_idx = current_pred_count
                    recent_preds_latent = predictions_buffer_latent[start_idx:end_idx]
                    y['y'] = recent_preds_latent[:, 0].reshape(
                        cond_time * latent_c, latent_h, latent_w
                    )
                else:
                    initial_conds = y['y'].reshape(cond_time, latent_c, latent_h, latent_w).to(self.device)
                    n_initial_needed = cond_time - current_pred_count
                    
                    combined = torch.zeros(
                        (cond_time, latent_c, latent_h, latent_w),
                        dtype=torch.float32,
                        device=self.device
                    )
                    
                    start_initial = cond_time - n_initial_needed
                    combined[:n_initial_needed] = initial_conds[start_initial:]
                    combined[n_initial_needed:] = predictions_buffer_latent[:current_pred_count, 0]
                    
                    y['y'] = combined.reshape(cond_time * latent_c, latent_h, latent_w)
                
                # Sample with the SAME encoded y
                # (y['y'] is updated but encoding is NOT called again)
                x_pred_latent = self.sample(
                    nsamples=nsamples,
                    shape=latent_shape,
                    y=y,
                    guidance=guidance,
                    nsteps=nsteps_diffusion,
                    record_history=False,
                    is_latent_shape=True,
                    return_in_latent_space=True,
                )
                
                next_idx = (step + 1) % max_buffer_size
                predictions_buffer_latent[next_idx] = x_pred_latent
                
                forecasts_latent.append(x_pred_latent)
            
            # Stack all latent predictions
            forecasts_latent_stacked = torch.stack(forecasts_latent, dim=0)
            
            if return_in_latent:
                return {
                    'forecasts': forecasts_latent_stacked,
                    'final_forecast_latent': forecasts_latent_stacked[-1],
                }
            
            # Decode all at once
            nsteps = forecasts_latent_stacked.shape[0]
            forecasts_latent_flat = forecasts_latent_stacked.view(
                nsteps * nsamples, latent_c, latent_h, latent_w
            )
            
            forecasts_pixel_flat = self.decode(forecasts_latent_flat, y, record_history=False)
            
            if isinstance(forecasts_pixel_flat, tuple):
                forecasts_pixel_flat = forecasts_pixel_flat[0]
            
            forecasts_pixel = forecasts_pixel_flat.view(
                nsteps, nsamples, *forecasts_pixel_flat.shape[1:]
            )
            
            result = {
                'forecasts': forecasts_pixel,
                'final_forecast': forecasts_pixel[-1],
            }
            
            if return_intermediate:
                result['intermediate_latent'] = forecasts_latent_stacked
            
            return result

    def _encode_y_once(self, y: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encode y dict ONCE. VAE sampling happens here."""
        if not hasattr(self, 'encode_y') or not self.encode_y:
            return y
        
        if 'y' not in y:
            return y
        
        dummy_x = torch.zeros(1, 3, 128, 128, device=self.device)
        
        try:
            encode_result = self.encode(dummy_x, y, record_history=False)
            if isinstance(encode_result, tuple):
                _, y_encoded = encode_result
                y_result = dict(y)
                y_result.update(y_encoded)
                if y_result['y'].shape[0] == 1:
                    y_result['y'] = y_result['y'].squeeze(0)
                return y_result
            else:
                return y
        except Exception as e:
            print(f"Warning: encoding y failed with {e}, using original y")
            return y

    def _autoregressive_sample_batched(
        self,
        nsamples: int,
        latent_shape: List[int],
        nsteps_forecast: int,
        cond_time: int,
        nsteps_diffusion: int,
        y: Optional[Dict[str, torch.Tensor]],
        y_already_encoded: bool,
        guidance: float,
        maximum_batch_size: int,
        return_intermediate: bool,
        return_in_latent: bool,
    ) -> Dict[str, torch.Tensor]:
        """Handle batching for memory efficiency."""
        batch_sizes = self._get_minibatch_sizes(nsamples, maximum_batch_size)
        
        results_list = []
        for batch_size in batch_sizes:
            result = self.autoregressive_sample(
                batch_size,
                latent_shape,
                nsteps_forecast,
                cond_time,
                nsteps_diffusion,
                y,
                y_already_encoded,
                guidance,
                maximum_batch_size=None,
                return_intermediate=return_intermediate,
                return_in_latent=return_in_latent,
            )
            results_list.append(result)
        
        forecasts = torch.cat([r['forecasts'] for r in results_list], dim=1)
        final_forecast = torch.cat([r['final_forecast'] for r in results_list], dim=0) \
            if 'final_forecast' in results_list[0] else None
        
        result = {'forecasts': forecasts}
        if final_forecast is not None:
            result['final_forecast'] = final_forecast
        if return_intermediate and 'intermediate_latent' in results_list[0]:
            result['intermediate_latent'] = torch.cat(
                [r['intermediate_latent'] for r in results_list], dim=1
            )
        
        return result

    def _get_minibatch_sizes(self, total: int, max_size: int) -> List[int]:
        """Compute minibatch sizes for batched processing."""
        nbatches = (total + max_size - 1) // max_size
        base_size = total // nbatches
        remainder = total % nbatches
        return [base_size + (1 if i < remainder else 0) for i in range(nbatches)]


# ============================================================================
# USAGE PATTERN - THE KEY FIX
# ============================================================================

# OLD (WRONG - encodes y multiple times):
#   result = model.autoregressive_sample(
#       nsamples=4,
#       latent_shape=[4, 32, 32],
#       nsteps_forecast=10,
#       cond_time=4,
#       y=y_pixel,  # Pixel space
#   )
#   # autoregressive_sample calls _encode_y_once() → VAE sample 1
#   # Gets different encoding than direct sample() → VAE sample 2

# NEW (CORRECT - encode once, reuse):
#   # Encode y ONCE
#   y_encoded = model._encode_y_once(y_pixel)
#
#   # Use same encoded y for both methods
#   result_auto = model.autoregressive_sample(
#       nsamples=4,
#       latent_shape=[4, 32, 32],
#       nsteps_forecast=10,
#       cond_time=4,
#       y=y_encoded,
#       y_already_encoded=True,  # Skip re-encoding
#   )
#
#   result_direct = model.sample(
#       nsamples=4,
#       shape=[4, 32, 32],
#       y=y_encoded,
#       is_latent_shape=True,
#       return_in_latent_space=True,
#   )
#
#   # Now both use SAME encoded y → similar results!