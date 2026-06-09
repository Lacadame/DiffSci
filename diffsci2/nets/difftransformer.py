import torch
import einops
from torch import Tensor
from jaxtyping import Float

from .commonlayers import GaussianFourierProjection


def patchfy(x: Float[Tensor, "b c h w"],  # noqa: F722
            patch_size: int):
    return einops.rearrange(x, 'b c (h p1) (w p2) -> b (h w) c p1 p2',
                            p1=patch_size,
                            p2=patch_size)


def unpatchfy(x, patch_size, H, W):
    h, w = H//patch_size, W//patch_size
    return einops.rearrange(x, 'b (h w) c p1 p2 -> b c (h p1) (w p2)',
                            p1=patch_size,
                            p2=patch_size, h=h, w=w)


def adaln_modulate(x, shift, scale):
    # x : (b, s, d)
    # shift : (b, d)
    # scale : (b, d)
    # returns : (b, s, d)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ResnetTimeBlock(torch.nn.Module):
    def __init__(self, embed_channels):

        """
        Parameters
        ----------
        embed_channels : int
            The number of channels in the embedding
        ouput_channels : int
            The number of channels in the output

        """

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embed_channels, 4*embed_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(4*embed_channels, 4*embed_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(4*embed_channels, embed_channels)
        )

    def forward(self, te):

        """
        Parameters
        ----------
        te : torch.Tensor of shape (nbatch, embed_channels)

        Returns
        -------
        torch.Tensor of shape (nbatch, output_channels, 1, 1)
        """

        # te : (nbatch, embed_channels)
        # returns : (nbatch, embed_channels)
        return te + self.net(te)


class Patcher(torch.nn.Module):
    def __init__(self, patch_size: int,
                 C: int):
        super().__init__()
        self.C = C
        self.patch_size = patch_size

    def forward(self, x):
        x = self.forward_unflatttened(x)
        x = einops.rearrange(x, 'b s c p1 p2 -> b s (c p1 p2)')
        return x

    def forward_unflatttened(self, x):
        x = patchfy(x, self.patch_size)
        return x

    def inverse(self, x, H, W):
        x = einops.rearrange(x, 'b s (c p1 p2) -> b s c p1 p2',
                             p1=self.patch_size,
                             p2=self.patch_size,
                             c=self.C)
        return self.inverse_unflattened(x, H, W)

    def inverse_unflattened(self, x, H, W):
        return unpatchfy(x, self.patch_size, H, W)


class PositionalEncoding2d(torch.nn.Module):
    def __init__(self, dembed, denominator=10000.0):
        super().__init__()
        self.dembed = dembed
        self.denominator = denominator
        dembed1d = dembed // 2
        indexes = torch.arange(start=0, end=dembed1d, step=2)  # [dembed//4]
        div_term = denominator ** (indexes / dembed1d)  # [dembed//4]
        self.register_buffer('div_term', div_term)

    def forward(self, h, w):
        w_indexes = torch.arange(h, dtype=torch.float32).repeat_interleave(w)
        h_indexes = torch.arange(w, dtype=torch.float32).repeat(h)
        sin_cos_w = self.encode(w_indexes)
        sin_cos_h = self.encode(h_indexes)
        sin_cos = torch.cat([sin_cos_w, sin_cos_h], axis=-1)
        return sin_cos

    def encode(self, x):
        # x : [seq_len]
        sin = torch.sin(x.unsqueeze(-1)/self.div_term)  # [seq_len, dembed//2]
        cos = torch.cos(x.unsqueeze(-1)/self.div_term)  # [seq_len, dembed//2]
        sin_cos = (torch.stack([sin, cos], axis=-1).
                   flatten(start_dim=-2))  # [seq_len, dembed]
        return sin_cos


class SelfAttention(torch.nn.Module):
    def __init__(self,
                 nembed,
                 nheads):
        super().__init__()
        self.nembed = nembed
        self.nheads = nheads
        self.attn = torch.nn.MultiheadAttention(nembed,
                                                nheads,
                                                batch_first=True)

    def forward(self, x):
        return self.attn(x, x, x)[0]


class DiTBlock(torch.nn.Module):
    def __init__(self,
                 nembed,
                 nheads,
                 mlp_factor=4):
        super().__init__()
        self.nembed = nembed
        self.nheads = nheads
        self.nmlp = mlp_factor*nembed
        self.norm1 = torch.nn.LayerNorm(nembed)
        self.norm2 = torch.nn.LayerNorm(nembed)
        self.attn = SelfAttention(nembed,
                                  nheads)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(nembed, self.nmlp),
            torch.nn.SiLU(),
            torch.nn.Linear(self.nmlp, nembed)
        )
        self.adaln_modulation = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(nembed, 6*nembed)
        )

    def forward(self, x, c):
        (shift_msa,
         scale_msa,
         gate_msa,
         shift_mlp,
         scale_mlp,
         gate_mlp) = self.adaln_modulation(c).chunk(6, dim=1)
        x = (x +
             gate_msa.unsqueeze(1) * self.attn(
                 adaln_modulate(self.norm1(x), shift_msa, scale_msa)))
        x = (x +
             gate_mlp.unsqueeze(1) * self.mlp(
                 adaln_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


class DiTCore(torch.nn.Module):
    def __init__(self,
                 nembed,
                 nheads,
                 nblocks,
                 mlp_factor=4):
        super().__init__()
        self.nembed = nembed
        self.nheads = nheads
        self.mlp_factor = mlp_factor
        self.nblocks = nblocks
        self.blocks = torch.nn.ModuleList(
            [DiTBlock(nembed,
                      nheads,
                      mlp_factor) for _ in range(nblocks)])

    def forward(self, x, c):
        for block in self.blocks:
            x = block(x, c)
        return x


class DiffusionTransformer(torch.nn.Module):
    def __init__(self,
                 nembed=64,
                 nheads=4,
                 mlp_factor=4,
                 nblocks=6,
                 patch_size=4,
                 nchannels=1):
        super().__init__()
        self.nembed = nembed
        self.nheads = nheads
        self.mlp_factor = mlp_factor
        self.nblocks = nblocks
        self.patch_size = patch_size
        self.nchannels = nchannels
        self.patcher = Patcher(patch_size, nchannels)
        self.core = DiTCore(nembed,
                            nheads,
                            mlp_factor,
                            nblocks)
        self.positional_encoding = PositionalEncoding2d(nembed)
        self.embed = torch.nn.Linear(nchannels*patch_size**2, nembed)
        self.unembed = torch.nn.Linear(nembed, nchannels*patch_size**2)
        self.time_embed = GaussianFourierProjection(nembed)
        self.resnet_time_block = ResnetTimeBlock(nembed)

    def forward(self, x, t):
        # x : [b, c, h, w]
        # t : [t]
        H, W = x.shape[-2], x.shape[-1]
        te = self.resnet_time_block(self.time_embed(t))  # [b, nembed]
        x = self.patcher(x)  # [b, s, c p1 p2]
        x = self.embed(x)  # [b, s, nembed]
        x = self.core(x, te)  # [b, s, nembed]
        x = self.unembed(x)  # [b, s, c p1 p2]
        x = self.patcher.inverse(x, H, W)  # [b, c, h, w]
        return x
