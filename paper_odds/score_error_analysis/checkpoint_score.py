"""Inference-only loader for the 144 default3/4/5 mixture checkpoints.

Matches Improved1DMLP and CustomEDMPreconditioner(mode='default') in
scripts/test-time_profile-correlation.py, without Lightning/diffusers imports.
Checkpoint files are local trusted training artifacts, including optimizer state.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        # Match the training implementation's float32 frequency constants.
        frequencies = torch.exp(torch.arange(half_dim, device=t.device)
                                * -(math.log(10000)/(half_dim-1)))
        phase = t[:, None] * frequencies[None, :]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


class MixtureMLP(nn.Module):
    def __init__(self, data_dim=1, time_embed_dim=64, hidden_dim=128, num_layers=3):
        super().__init__()
        self.time_embed = SinusoidalEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_embed_dim, hidden_dim), nn.SiLU(),
                                      nn.Linear(hidden_dim, hidden_dim))
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                          nn.Linear(hidden_dim, hidden_dim)) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, data_dim)
        self.act = nn.SiLU()

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time_embed(t))
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(self.act(h+t_emb))  # residual=False in these runs
        return self.output_proj(h)


class CheckpointScore:
    def __init__(self, path, device='cpu', dtype=torch.float64):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.epoch = int(checkpoint['epoch'])
        state = checkpoint['state_dict']
        if any(not k.startswith('model.') for k in state):
            raise ValueError("Unexpected checkpoint buffers; inspect preconditioning before loading.")
        self.model = MixtureMLP()
        self.model.load_state_dict({k.removeprefix('model.'): v for k, v in state.items()}, strict=True)
        self.model = self.model.to(device=device, dtype=dtype).eval()
        self.device, self.dtype = device, dtype

    @torch.inference_mode()
    def __call__(self, x, sigma, batch_size=8192):
        x, sigma = np.broadcast_arrays(np.asarray(x), np.asarray(sigma))
        if not np.isfinite(x).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError("Inference requires finite x and positive finite sigma.")
        outputs = []
        for start in range(0, x.size, batch_size):
            xs = torch.as_tensor(x.ravel()[start:start+batch_size].copy(), device=self.device, dtype=self.dtype)[:, None]
            sig = torch.as_tensor(sigma.ravel()[start:start+batch_size].copy(), device=self.device, dtype=self.dtype)
            denom = torch.sqrt(sig**2+0.5**2)
            raw = self.model(xs/denom[:, None], 0.5*torch.log(sig))
            # Algebraically equal to (denoiser-x)/sigma², avoiding cancellation.
            score = 0.5*raw/(sig*denom)[:, None] - xs/(denom**2)[:, None]
            outputs.append(score[:, 0].cpu().numpy())
        return np.concatenate(outputs).reshape(x.shape)
