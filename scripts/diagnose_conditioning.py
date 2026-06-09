#!/usr/bin/env python
"""
Diagnostic script to analyze conditioning in existing models.

Helps identify:
1. Whether cond_drop is enabled (required for CFG)
2. Magnitude of conditioning vs time embeddings
3. Test enhanced conditioning wrapper

Usage:
    python scripts/diagnose_conditioning.py --checkpoint /path/to/model.ckpt
"""

import argparse
import sys
import os

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'exploratory', 'dfn', 'aux'))

import diffsci2.models
from diffsci2.nets.enhanced_conditioning import wrap_model_with_enhanced_conditioning
from model_loaders import load_flow_model, load_autoencoder


def analyze_model(checkpoint_path: str, stone: str = 'Estaillades'):
    """Analyze conditioning configuration of a model."""

    print("=" * 70)
    print("Model Conditioning Diagnostic")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print()

    # Load model
    if checkpoint_path.lower() == 'base':
        print("Loading base (pretrained) model...")
        model = load_flow_model(stone)
    else:
        print("Loading from checkpoint...")
        model = load_flow_model(checkpoint_path, custom_checkpoint_path=True)

    print(f"Model type: {type(model).__name__}")
    print()

    # Check conditioning configuration
    print("=" * 70)
    print("Conditioning Configuration")
    print("=" * 70)

    config = model.config
    print(f"  model_channels: {config.model_channels}")
    print(f"  cond_dropout: {config.cond_dropout}")
    print(f"  cond_drop: {config.cond_drop}")
    print(f"  cond_drop_learnable: {config.cond_drop_learnable}")
    print()

    # Check if cond_drop module exists
    print("Classifier-Free Guidance Status:")
    if hasattr(model, 'cond_drop') and model.cond_drop is not None:
        print(f"  cond_drop module: ENABLED")
        print(f"  cond_drop.p: {model.cond_drop.p}")
        if hasattr(model.cond_drop, 'null_embedding'):
            print(f"  null_embedding: {model.cond_drop.null_embedding.shape}")
            print(f"  null_embedding mean: {model.cond_drop.null_embedding.data.mean():.4f}")
    else:
        print(f"  cond_drop module: DISABLED (None)")
        print()
        print("  ⚠️  WARNING: CFG not enabled during training!")
        print("  ⚠️  Using guidance > 1.0 at inference won't help.")
        print("  ⚠️  Consider retraining with cond_drop > 0")
    print()

    # Check conditional embedding
    print("Conditional Embedding:")
    if model.conditional_embedding is not None:
        print(f"  Type: {type(model.conditional_embedding).__name__}")
        print(f"  Params: {sum(p.numel() for p in model.conditional_embedding.parameters()):,}")
    else:
        print(f"  None (no conditional embedding)")
    print()

    # Test forward pass with dummy data
    print("=" * 70)
    print("Embedding Magnitude Analysis")
    print("=" * 70)

    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # Create dummy inputs
    batch_size = 1
    latent_size = 32
    x = torch.randn(batch_size, 4, latent_size, latent_size, latent_size, device=device)
    t = torch.tensor([0.5], device=device)  # Middle timestep
    porosity = torch.rand(latent_size, latent_size, latent_size, device=device) * 0.5 + 0.1  # 0.1-0.6 range
    y = {'porosity': porosity}

    with torch.no_grad():
        # Get time embedding
        te = model.time_projection(t)
        print(f"Time embedding (t=0.5):")
        print(f"  Shape: {te.shape}")
        print(f"  Mean: {te.mean():.4f}")
        print(f"  Std: {te.std():.4f}")
        print(f"  Abs max: {te.abs().max():.4f}")
        print()

        # Get conditional embedding
        if model.conditional_embedding is not None:
            ye = model.conditional_embedding(y)
            print(f"Condition embedding (porosity ~0.35):")
            print(f"  Shape: {ye.shape}")
            print(f"  Mean: {ye.mean():.4f}")
            print(f"  Std: {ye.std():.4f}")
            print(f"  Abs max: {ye.abs().max():.4f}")
            print()

            # Compare magnitudes
            if ye.ndim == te.ndim:
                ratio = ye.std() / te.std()
            else:
                ratio = ye.std() / te.std()

            print(f"Condition/Time std ratio: {ratio:.4f}")
            if ratio < 0.5:
                print("  ⚠️  Conditioning signal is weak relative to time")
                print("  ⚠️  Consider amplifying conditioning")
            elif ratio > 2.0:
                print("  ✓  Conditioning signal is strong relative to time")
            else:
                print("  ✓  Conditioning and time signals are balanced")
    print()

    # Test enhanced wrapper
    print("=" * 70)
    print("Testing Enhanced Conditioning Wrapper")
    print("=" * 70)

    try:
        enhanced = wrap_model_with_enhanced_conditioning(
            model=model,
            use_film=True,
            use_multiscale=True,
            use_gradient=True,
            cond_drop_p=0.1,
        )

        base_params = sum(p.numel() for p in enhanced.base_model.parameters())
        new_params = sum(p.numel() for p in enhanced.get_trainable_params())

        print(f"  Wrapper created successfully!")
        print(f"  Base params: {base_params:,}")
        print(f"  New params: {new_params:,}")
        print(f"  Overhead: {100 * new_params / base_params:.1f}%")
        print()

        # Test forward pass
        enhanced = enhanced.to(device)
        with torch.no_grad():
            out = enhanced(x, t, y)
        print(f"  Forward pass successful!")
        print(f"  Output shape: {out.shape}")
        print(f"  Output mean: {out.mean():.4f}")
        print(f"  Output std: {out.std():.4f}")

    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 70)
    print("Recommendations")
    print("=" * 70)

    recommendations = []

    if model.cond_drop is None:
        recommendations.append(
            "1. CRITICAL: Train with cond_drop > 0 for CFG to work.\n"
            "   Use: --cond-drop-p 0.1 with 0003b-porosity-field-training-enhanced.py"
        )

    recommendations.append(
        "2. After training with enhanced conditioning, try:\n"
        "   --guidance 1.5, 2.0, 3.0 at inference"
    )

    recommendations.append(
        "3. Monitor FiLM layer weights during training:\n"
        "   - gamma/beta should start near zero\n"
        "   - Should learn non-zero values"
    )

    for i, rec in enumerate(recommendations):
        print(rec)
        print()

    return model


def main():
    parser = argparse.ArgumentParser(description='Diagnose conditioning in models')
    parser.add_argument('--checkpoint', type=str, default='base',
                        help='Path to checkpoint or "base" for pretrained')
    parser.add_argument('--stone', type=str, default='Estaillades',
                        choices=['Bentheimer', 'Doddington', 'Estaillades', 'Ketton'])
    args = parser.parse_args()

    analyze_model(args.checkpoint, args.stone)


if __name__ == '__main__':
    main()
