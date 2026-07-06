"""
Conversion script from AutoencoderKL (old) to VAENet (new) architecture.

This script handles the reorganization of quant_conv and post_quant_conv layers
from top-level to being embedded within encoder and decoder respectively.
"""

import torch
import yaml
from pathlib import Path
from typing import Dict, Any


def convert_state_dict(old_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert state dict from AutoencoderKL format to VAENet format.
    
    Key mappings:
    - quant_conv.* -> encoder.quant_conv.conv.*
    - post_quant_conv.* -> decoder.post_quant_conv.conv.*
    - encoder.* -> encoder.* (with .conv insertion for Conv layers)
    - decoder.* -> decoder.* (with .conv insertion for Conv layers)
    - loss.* -> skip (not needed)
    
    The new model uses PatchedConv which wraps Conv3d in a .conv attribute,
    so we need to insert .conv before .weight and .bias for all Conv layers.
    """
    new_state_dict = {}
    
    for key, value in old_state_dict.items():
        # Skip loss module weights
        if key.startswith('loss.'):
            continue
            
        # Map quant_conv to encoder.quant_conv
        if key.startswith('quant_conv.'):
            new_key = 'encoder.' + key
            # Insert .conv before weight/bias
            new_key = new_key.replace('.weight', '.conv.weight').replace('.bias', '.conv.bias')
            new_state_dict[new_key] = value
            
        # Map post_quant_conv to decoder.post_quant_conv
        elif key.startswith('post_quant_conv.'):
            new_key = 'decoder.' + key
            # Insert .conv before weight/bias
            new_key = new_key.replace('.weight', '.conv.weight').replace('.bias', '.conv.bias')
            new_state_dict[new_key] = value
            
        # Keep encoder and decoder weights, but add .conv for Conv layers
        elif key.startswith('encoder.') or key.startswith('decoder.'):
            new_key = key
            
            # Special case: decoder upsample conv layers need double .conv
            # e.g., decoder.up.1.upsample.conv.weight -> decoder.up.1.upsample.conv.conv.weight
            # But encoder downsample conv stays as is (uses direct Conv3d, not PatchedConv)
            if key.startswith('decoder.') and 'upsample.conv.' in key:
                if key.endswith('.weight'):
                    new_key = key.replace('.weight', '.conv.weight')
                elif key.endswith('.bias'):
                    new_key = key.replace('.bias', '.conv.bias')
            # Encoder downsample conv stays as-is (already has .conv in path)
            elif key.startswith('encoder.') and 'downsample.conv.' in key:
                # Keep the key as-is, it's already correct
                new_key = key
            # Insert .conv before weight/bias for other Conv layers
            elif any(pattern in key for pattern in [
                'conv_in.', 'conv_out.', 'conv1.', 'conv2.', 
                'conv_shortcut.', 'nin_shortcut.',
                '.q.', '.k.', '.v.', '.proj_out.'  # Attention layers if present
            ]):
                # Don't double-add .conv if it's already there somehow
                if '.conv.weight' not in key and '.conv.bias' not in key:
                    new_key = new_key.replace('.weight', '.conv.weight').replace('.bias', '.conv.bias')
            
            new_state_dict[new_key] = value
            
        else:
            print(f"Warning: Skipping unexpected key: {key}")
    
    return new_state_dict


def load_old_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load old AutoencoderKL checkpoint."""
    print(f"Loading old checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle both direct state_dict and Lightning checkpoint formats
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    
    print(f"Loaded {len(state_dict)} keys from old checkpoint")
    return state_dict


def create_new_model(config_dict: Dict[str, Any]):
    """Create new VAENet model from config."""
    from diffsci.models.nets.vaenet import VAENet, VAENetConfig
    
    config = VAENetConfig(**config_dict)
    model = VAENet(config)
    print(f"Created new VAENet model")
    return model


def verify_conversion(old_state_dict: Dict[str, torch.Tensor], 
                     new_state_dict: Dict[str, torch.Tensor],
                     new_model_state_dict: Dict[str, torch.Tensor]):
    """Verify that the conversion was successful."""
    print("\n=== Conversion Verification ===")
    
    # Count keys
    old_model_keys = [k for k in old_state_dict.keys() if not k.startswith('loss.')]
    print(f"Old model keys (excluding loss): {len(old_model_keys)}")
    print(f"New state dict keys: {len(new_state_dict)}")
    print(f"New model expected keys: {len(new_model_state_dict)}")
    
    # Check if all new model keys are present
    missing_keys = set(new_model_state_dict.keys()) - set(new_state_dict.keys())
    unexpected_keys = set(new_state_dict.keys()) - set(new_model_state_dict.keys())
    
    if missing_keys:
        print(f"\n❌ Missing keys ({len(missing_keys)}):")
        for key in sorted(missing_keys)[:10]:
            print(f"  - {key}")
        if len(missing_keys) > 10:
            print(f"  ... and {len(missing_keys) - 10} more")
    else:
        print("✓ No missing keys")
    
    if unexpected_keys:
        print(f"\n❌ Unexpected keys ({len(unexpected_keys)}):")
        for key in sorted(unexpected_keys)[:10]:
            print(f"  - {key}")
        if len(unexpected_keys) > 10:
            print(f"  ... and {len(unexpected_keys) - 10} more")
    else:
        print("✓ No unexpected keys")
    
    # Check shapes
    shape_mismatches = []
    for key in new_state_dict.keys():
        if key in new_model_state_dict:
            if new_state_dict[key].shape != new_model_state_dict[key].shape:
                shape_mismatches.append(
                    (key, new_state_dict[key].shape, new_model_state_dict[key].shape)
                )
    
    if shape_mismatches:
        print(f"\n❌ Shape mismatches ({len(shape_mismatches)}):")
        for key, old_shape, new_shape in shape_mismatches[:10]:
            print(f"  - {key}: {old_shape} vs {new_shape}")
    else:
        print("✓ All shapes match")
    
    success = not missing_keys and not unexpected_keys and not shape_mismatches
    if success:
        print("\n✓✓✓ Conversion successful! ✓✓✓")
    else:
        print("\n❌ Conversion has issues that need to be addressed")
    
    return success


def convert_checkpoint(
    old_checkpoint_path: str,
    output_path: str,
    new_model_config: Dict[str, Any],
    verify: bool = True
) -> bool:
    """
    Main conversion function.
    
    Args:
        old_checkpoint_path: Path to old AutoencoderKL checkpoint
        output_path: Path to save converted checkpoint
        new_model_config: Configuration dict for VAENet
        verify: Whether to verify the conversion
        
    Returns:
        True if conversion successful, False otherwise
    """
    # Load old checkpoint
    old_state_dict = load_old_checkpoint(old_checkpoint_path)
    
    # Convert state dict
    print("\nConverting state dict...")
    new_state_dict = convert_state_dict(old_state_dict)
    
    # Create new model for verification
    new_model = create_new_model(new_model_config)
    
    # Verify if requested
    if verify:
        success = verify_conversion(
            old_state_dict, 
            new_state_dict, 
            new_model.state_dict()
        )
        if not success:
            print("\n⚠️  Conversion verification failed. Not saving checkpoint.")
            return False
    
    # Load weights into model to ensure they work
    print("\nLoading converted weights into model...")
    try:
        new_model.load_state_dict(new_state_dict, strict=True)
        print("✓ Weights loaded successfully")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return False
    
    # Save converted checkpoint
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving converted checkpoint to: {output_path}")
    torch.save({
        'state_dict': new_state_dict,
        'config': new_model_config,
        'conversion_info': {
            'original_checkpoint': old_checkpoint_path,
            'architecture': 'VAENet',
        }
    }, output_path)
    print("✓ Checkpoint saved")
    
    return True


def main():
    """Example usage with standard configuration."""
    
    # Standard configuration matching the YAML
    config = {
        'dimension': 3,
        'in_channels': 1,
        'out_channels': 1,
        'z_channels': 4,
        'z_dim': 4,
        'ch': 32,
        'ch_mult': [1, 2, 4, 4],
        'num_res_blocks': 2,
        'attn_resolutions': [],
        'dropout': 0.0,
        'resolution': 256,
        'has_mid_attn': False,
        'resamp_with_conv': True,
        'attn_type': 'vanilla',
        'tanh_out': False,
        'input_bias': True,
        'output_bias': True,
        'with_time_emb': False,
        'double_z': True,
        'num_groups': 32,
        'patch_size': None,
        'memory_efficient_variant': False,
        'use_flash_attention': True,
        'minimal_rf_mode': False,
    }
    
    # Paths from the YAML config
    old_checkpoint = '/home/ubuntu/repos/PoreGen/savedmodels/production/20241029-bps-ldmvae-4imperial/checkpoints/model-epoch=060-val_loss=0.002525.ckpt'
    output_checkpoint = '/home/ubuntu/repos/DiffSci/notebooks/exploratory/dfn/converted_vaenet.ckpt'
    
    # Run conversion
    success = convert_checkpoint(
        old_checkpoint_path=old_checkpoint,
        output_path=output_checkpoint,
        new_model_config=config,
        verify=True
    )
    
    if success:
        print("\n" + "="*60)
        print("Conversion completed successfully!")
        print(f"Converted checkpoint saved to: {output_checkpoint}")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Conversion failed. Please check the errors above.")
        print("="*60)


if __name__ == '__main__':
    main()

