"""Convert a `VAESFTModule` Lightning checkpoint into the "raw" pixnorm
ckpt layout (`encdec.`-prefixed, bare convs, no regressor).

The two SFT flavors:

| variant            | encoder/decoder keys on disk      | conversion |
|--------------------|-----------------------------------|------------|
| `vae_pixnorm_s4_sft` | PatchedConv (`<...>.conv.weight`) | strip `.conv.` infix + add `encdec.` prefix |
| `vae_pixnorm_s8_sft` | bare (`<...>.weight`)             | just add `encdec.` prefix |

In both cases the regressor and any `loss_module.*` keys are dropped.

The output file can be loaded by any code path that already handles the
"raw" pixnorm ckpts (`vae_pixnorm_s4_raw.ckpt` / `vae_pixnorm_s8_raw.ckpt`) —
including `diffsci2.vaesft.load_autoencoder(path=...)`.

Usage:
    python -m diffsci2.vaesft.ckpt_conversion \
        --in  /path/to/run03_pixnorm/best-step1400-z0.0732.ckpt \
        --out /path/to/run03_pixnorm/converted_pixnorm_sft.ckpt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


# Layer names wrapped by `PatchedConv` inside the main-package `VAENet`. On
# the SFT ckpt for s4 they show up as `<name>.conv.weight`; the bare
# layout has `<name>.weight`.
_PATCHED_NAMES = (
    "conv_in", "conv_out", "conv1", "conv2", "nin_shortcut",
    "quant_conv", "post_quant_conv",
)


def _is_s4_layout(state: dict) -> bool:
    """SFT-s4 has PatchedConv keys (`encoder.conv_in.conv.weight`).
    SFT-s8 has bare keys (`encoder.conv_in.weight`)."""
    return any("conv_in.conv.weight" in k for k in state)


def _sft_s4_to_pixnorm_raw_state(sft_state: dict) -> dict:
    """Inverse of `diffsci2.extra.legacy_loaders.remap_pixnorm_raw_state`.

    Takes a SFT state_dict (PatchedConv layout under `encoder.`/`decoder.`)
    and returns the bare-conv `encdec.`-prefixed layout. Drops everything
    outside encoder/decoder.
    """
    out: dict = {}
    for k, v in sft_state.items():
        if not (k.startswith("encoder.") or k.startswith("decoder.")):
            continue
        # Collapse PatchedConv `.conv` indirection for the named layers.
        for name in _PATCHED_NAMES:
            for suffix in ("weight", "bias"):
                k = k.replace(f".{name}.conv.{suffix}", f".{name}.{suffix}")
        # Decoder's `upsample.conv` is itself a PatchedConv; collapse the
        # extra `.conv`.
        for suffix in ("weight", "bias"):
            k = k.replace(f".upsample.conv.conv.{suffix}",
                          f".upsample.conv.{suffix}")
        # Rename layers that differ between the two layouts.
        k = k.replace(".nin_shortcut.", ".shortcut.")
        k = k.replace(".mid.block_1.", ".mid_block_1.")
        k = k.replace(".mid.block_2.", ".mid_block_2.")
        out[f"encdec.{k}"] = v
    return out


def _sft_s8_to_pixnorm_raw_state(sft_state: dict) -> dict:
    """SFT-s8 layout is already bare convs under `encoder.`/`decoder.`.
    Just filter to those keys and prepend `encdec.`."""
    out: dict = {}
    for k, v in sft_state.items():
        if not (k.startswith("encoder.") or k.startswith("decoder.")):
            continue
        out[f"encdec.{k}"] = v
    return out


def sft_to_pixnorm_raw_state(sft_state: dict) -> dict:
    """Dispatch on SFT ckpt layout (s4 PatchedConv vs s8 bare)."""
    if _is_s4_layout(sft_state):
        return _sft_s4_to_pixnorm_raw_state(sft_state)
    return _sft_s8_to_pixnorm_raw_state(sft_state)


def _round_trip_verify(converted_path: Path, original_sft_state: dict) -> None:
    """Load the converted ckpt back through `diffsci2.vaesft.load_autoencoder`
    and confirm the resulting VAE's state_dict matches the source SFT's
    encoder/decoder tensors exactly."""
    from diffsci2.vaesft import load_autoencoder
    vae = load_autoencoder(path=str(converted_path))
    loaded = vae.state_dict()

    enc_dec_keys = [k for k in original_sft_state
                    if k.startswith("encoder.") or k.startswith("decoder.")]
    mismatched, missing = [], []
    for k in enc_dec_keys:
        if k not in loaded:
            missing.append(k)
            continue
        if not torch.equal(loaded[k], original_sft_state[k]):
            mismatched.append(k)
    print(f"  source SFT enc/dec tensors : {len(enc_dec_keys)}")
    print(f"  loaded VAE tensors         : {len(loaded)}")
    print(f"  missing from loaded        : {len(missing)}")
    print(f"  bit-exact mismatch         : {len(mismatched)}")
    if missing:
        raise SystemExit(
            f"round-trip FAILED: {len(missing)} missing keys "
            f"(first: {missing[:3]})"
        )
    if mismatched:
        raise SystemExit(
            f"round-trip FAILED: {len(mismatched)} mismatched keys "
            f"(first: {mismatched[:3]})"
        )
    print("  -> round-trip OK (bit-exact)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", type=str, required=True)
    p.add_argument("--out", dest="out_path", type=str, required=True)
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()

    in_path = Path(args.in_path).resolve()
    out_path = Path(args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ckpt_conversion] reading {in_path}")
    blob = torch.load(in_path, map_location="cpu", weights_only=False)
    sft_state = blob["state_dict"]
    print(f"[ckpt_conversion] source state_dict tensors: {len(sft_state)}  "
          f"(encoder {sum(k.startswith('encoder.') for k in sft_state)}, "
          f"decoder {sum(k.startswith('decoder.') for k in sft_state)}, "
          f"regressor {sum(k.startswith('regressor.') for k in sft_state)})")
    layout = "s4 (PatchedConv)" if _is_s4_layout(sft_state) else "s8 (bare)"
    print(f"[ckpt_conversion] detected layout: {layout}")

    new_state = sft_to_pixnorm_raw_state(sft_state)
    print(f"[ckpt_conversion] converted tensors: {len(new_state)} (prefixed `encdec.`)")

    out_blob = {
        "state_dict": new_state,
        "source_sft_ckpt": str(in_path),
        "conversion_info": (
            "Generated by diffsci2.vaesft.ckpt_conversion. The state_dict "
            "follows the bare pixnorm layout (encdec.-prefixed). Loadable "
            "via diffsci2.vaesft.load_autoencoder(path=...) or any code "
            "path that accepts converted_vaenet_pixnorm.ckpt."
        ),
    }
    print(f"[ckpt_conversion] writing {out_path}")
    torch.save(out_blob, out_path)

    if not args.no_verify:
        print("[ckpt_conversion] round-trip verification:")
        _round_trip_verify(out_path, sft_state)


if __name__ == "__main__":
    main()
