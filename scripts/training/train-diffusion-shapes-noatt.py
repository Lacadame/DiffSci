import math
import os

import lightning
import lightning.pytorch.callbacks as pl_callbacks
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.draw import disk, polygon, rectangle
from torch.utils.data import Dataset

import diffsci.models


class ShapesDataset(Dataset):
    def __init__(
        self,
        size=64,
        length=10000,
        mode="paper_replica",
        polygon_size=8,
        seed=0,
        index_offset=0,
    ):
        """
        mode="paper_replica":
            Reproduces Aithal et al. (2024).
            3 implied columns. Slot 1: Triangle, Slot 2: Square, Slot 3: Pentagon.
            Each appears with 50% prob, and the height is sampled from a uniform distribution.

        mode="geometry_test":
            For testing your 'Squircle' hypothesis.
            Single centered object. Either a Square OR a Circle (50/50).
            This forces the network to interpolate *geometry*, not just *presence*.

        seed: Base random seed for reproducibility. Sample at idx uses seed + index_offset + idx.
        index_offset: Added to idx when seeding. Use different offsets for disjoint train/val
            (e.g. train: offset=0, length=8000; val: offset=8000, length=2000).
        """
        self.size = size
        self.length = length
        self.mode = mode
        self.polygon_size = polygon_size
        self.seed = seed
        self.index_offset = index_offset

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Deterministic per (idx + index_offset): same index always yields same sample
        sample_seed = self.seed + self.index_offset + idx
        np.random.seed(sample_seed)
        torch.manual_seed(sample_seed)

        img = np.zeros((self.size, self.size), dtype=np.float32)
        
        if self.mode == "paper_replica":
            # 3 columns: left, center, right. Each column has its own random height (row).
            heights = np.random.randint(
                self.size // 4, 3 * self.size // 4, size=3
            )  # row (y) per slot
            cols = [self.size // 6, self.size // 2, 5 * self.size // 6]  # column (x) positions
            shape_size = self.polygon_size

            # Slot 1: Triangle (50% prob), centered at (heights[0], cols[0])
            if torch.rand(1) > 0.5:
                rr, cc = polygon(
                    [heights[0] - shape_size, heights[0] + shape_size, heights[0] + shape_size],
                    [cols[0], cols[0] - shape_size, cols[0] + shape_size],
                    shape=img.shape,
                )
                img[rr, cc] = 1.0

            # Slot 2: Square (50% prob), centered at (heights[1], cols[1])
            if torch.rand(1) > 0.5:
                r, c = heights[1] - shape_size, cols[1] - shape_size
                rr, cc = rectangle(start=(r, c), extent=(shape_size * 2, shape_size * 2), shape=img.shape)
                img[rr, cc] = 1.0

            # Slot 3: Pentagon (Approximated as disk, 50% prob), centered at (heights[2], cols[2])
            if torch.rand(1) > 0.5:
                rr, cc = disk((heights[2], cols[2]), shape_size, shape=img.shape)
                img[rr, cc] = 1.0
                
        elif self.mode == "geometry_test":
            # Centered shapes for morphing test
            center = (self.size // 2, self.size // 2)
            radius = self.size // 4
            
            if torch.rand(1) > 0.5:
                # Square
                start = (center[0] - radius, center[1] - radius)
                rr, cc = rectangle(start=start, extent=(radius*2, radius*2), shape=img.shape)
            else:
                # Circle
                rr, cc = disk(center, radius, shape=img.shape)
            
            img[rr, cc] = 1.0

        # Transform to Tensor [1, H, W] and normalize to [-1, 1] for diffusion
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        return img_tensor * 2 - 1


# --- Helper Modules ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResBlock(nn.Module):
    """Standard ResBlock with Time Embedding Injection"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        )
        self.block1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.block2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm1 = nn.GroupNorm(32, in_channels) # Using GroupNorm as is standard in DDPM
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_emb):
        # Time Embedding Projection
        scale, shift = self.mlp(time_emb).chunk(2, dim=1)
        scale, shift = scale[:, :, None, None], shift[:, :, None, None]

        h = self.block1(F.silu(self.norm1(x)))
        h = h * (scale + 1) + shift
        h = self.block2(self.dropout(F.silu(self.norm2(h))))
        return h + self.res_conv(x)

class AttentionBlock(nn.Module):
    """
    The culprit? This block contains the MLP (FeedForward) 
    that might be introducing the spectral bias in feature space.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # --- THIS IS THE MLP YOU SUSPECT ---
        # A simple position-wise FeedForward network
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1).permute(0, 2, 1) # [B, L, C]
        
        # 1. Self-Attention
        norm_x = self.norm(x).view(b, c, -1).permute(0, 2, 1)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x_flat = x_flat + attn_out
        
        # 2. MLP (FeedForward)
        # If your hypothesis is correct, the spectral bias here forces 
        # 'x_flat' to change smoothly, preventing sharp transitions in semantics.
        x_flat = x_flat + self.ff(x_flat)
        
        return x_flat.permute(0, 2, 1).view(b, c, h, w)

# --- Main U-Net ---
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, base_dim=64, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        
        # Time Embedding
        time_dim = base_dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder (Downsampling)
        self.init_conv = nn.Conv2d(in_channels, base_dim, 3, padding=1)
        
        # Resolution: 64 -> 32 -> 16 -> 8
        self.downs = nn.ModuleList([
            ResBlock(base_dim, base_dim, time_dim),
            ResBlock(base_dim, base_dim*2, time_dim),
            ResBlock(base_dim*2, base_dim*4, time_dim),
        ])
        self.down_samples = nn.ModuleList([
            nn.Conv2d(base_dim, base_dim, 4, 2, 1),
            nn.Conv2d(base_dim*2, base_dim*2, 4, 2, 1),
            nn.Conv2d(base_dim*4, base_dim*4, 4, 2, 1),
        ])

        # Bottleneck (Resolution 8x8)
        self.mid_block1 = ResBlock(base_dim*4, base_dim*4, time_dim)
        
        if self.use_attention:
            self.mid_attn = AttentionBlock(base_dim*4)
        else:
            # Replace Attention with an extra ResBlock to keep depth/capacity similar
            self.mid_attn = ResBlock(base_dim*4, base_dim*4, time_dim)
            
        self.mid_block2 = ResBlock(base_dim*4, base_dim*4, time_dim)

        # Decoder (Upsampling)
        self.ups = nn.ModuleList([
            ResBlock(base_dim*8, base_dim*2, time_dim), # Concatenation doubles channels
            ResBlock(base_dim*4, base_dim, time_dim),
            ResBlock(base_dim*2, base_dim, time_dim),
        ])
        self.up_samples = nn.ModuleList([
            nn.ConvTranspose2d(base_dim*4, base_dim*4, 4, 2, 1),
            nn.ConvTranspose2d(base_dim*2, base_dim*2, 4, 2, 1),
            nn.ConvTranspose2d(base_dim, base_dim, 4, 2, 1),
        ])

        self.final_conv = nn.Conv2d(base_dim, in_channels, 3, padding=1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        x = self.init_conv(x)
        
        # Encoder
        skips = [x]
        for block, down in zip(self.downs, self.down_samples):
            x = block(x, t)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.mid_block1(x, t)
        if self.use_attention:
            x = self.mid_attn(x) # Attention + MLP
        else:
            x = self.mid_attn(x, t) # Pure ResBlock
        x = self.mid_block2(x, t)

        # Decoder
        for block, up in zip(self.ups, self.up_samples):
            x = up(x)
            skip = skips.pop()
            x = torch.cat((x, skip), dim=1)
            x = block(x, t)

        return self.final_conv(x)


def main():

    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Set parameters
    device_id = 7
    batch_size = 32
    model_channels = 64
    n_epochs = 50
    learning_rate = 1e-4
    checkpoint_dir = f"/home/ubuntu/repos/DiffSci/savedmodels/experimental/20260215-bps-shapes-noatt"

    # Load dataset (disjoint: train uses indices 0..train_len-1, val uses train_len..train_len+val_len-1)
    train_len = 8000
    val_len = 2000
    train_dataset = ShapesDataset(
        mode="paper_replica",
        polygon_size=6,
        length=train_len,
        index_offset=0,
    )
    val_dataset = ShapesDataset(
        mode="paper_replica",
        polygon_size=6,
        length=val_len,
        index_offset=train_len,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
    )


    # Define model
    model = SimpleUNet(base_dim=model_channels, use_attention=False)
    moduleconfig = diffsci.models.KarrasModuleConfig.from_edm()
    module = diffsci.models.KarrasModule(model, moduleconfig, conditional=False)

    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate)
    module.set_optimizer_and_scheduler(optimizer) 
    
    callbacks = [diffsci.models.callbacks.NanToZeroGradCallback()]

    # Create directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath=os.path.join(checkpoint_dir, 'checkpoints'),
        filename='model-{epoch:03d}-{val_loss:.6f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True
    )

    trainer = lightning.Trainer(
        accelerator="gpu",
        devices=[device_id],
        max_epochs=n_epochs,
        default_root_dir=checkpoint_dir,
        gradient_clip_val=0.5,
        callbacks=callbacks + [checkpoint_callback],
        enable_checkpointing=True
    )

    print(f"Starting training on device {device_id}, with batch size {batch_size} and learning rate {learning_rate} for {n_epochs} epochs.")

    trainer.fit(module, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()