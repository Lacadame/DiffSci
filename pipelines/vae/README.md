# pipelines/vae/

Operator scripts for the VAE supervised-fine-tuning pipeline. Each script
is a thin CLI wrapping the library code in `diffsci2.vaesft`.

| script | purpose |
|---|---|
| `train_vae_sft.py` | DDP training entrypoint. Builds `VAESFTModule`, runs Lightning trainer. |
| `eval_vae_sft.py` | Standalone held-out eval (CSV + npz) on the regressor cache. |
| `eval_pnm_256.py` | 256³ reconstruction + PNM-permeability comparison (per-stone CSVs + figures). |
| `smoke_test_sft.py` | Single-GPU smoke check (~5 steps + 1 val pass). |
| `paths.py` | Host-specific defaults (CKPT_DIR / LOG_DIR / PLOT_DIR / EXISTING_CSV_DIR). |

The library code these wrap lives at:

- `diffsci2.vaesft.VAESFTModule` / `SFTConfig` — Lightning module.
- `diffsci2.vaesft.load_autoencoder(...)` — VAE convenience loader.
- `diffsci2.vaesft.FrozenRegressor` — frozen perceptual-loss reward.

Invocation:

```bash
cd /home/ubuntu/repos/DiffSci2

# Smoke test
/opt/persistence/miniconda3/envs/ddpm_env/bin/python \
    pipelines/vae/smoke_test_sft.py

# Full training (5-way DDP, default config)
/opt/persistence/miniconda3/envs/ddpm_env/bin/python \
    pipelines/vae/train_vae_sft.py \
        --run-name run05 \
        --vae-variant vae_pixnorm_s8_raw \
        --devices 1,2,3,4,5 \
        --total-steps 10000

# Eval
/opt/persistence/miniconda3/envs/ddpm_env/bin/python \
    pipelines/vae/eval_vae_sft.py \
        --ckpt <path-to-best-ckpt>.ckpt \
        --n-samples 256

# 256^3 head-to-head
/opt/persistence/miniconda3/envs/ddpm_env/bin/python \
    pipelines/vae/eval_pnm_256.py \
        --ckpt <path-to-best-ckpt>.ckpt \
        --mode sft --run-name run05
```

The old equivalents under
`notebooks/exploratory/dfnai/scripts/vaeporesft/{train,evaluate,eval_pnm_256,smoke_test}.py`
are unchanged and still work — two paths coexist during the transition.
