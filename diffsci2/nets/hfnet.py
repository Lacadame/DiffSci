import torch
import diffusers

"""
A set of wrappers for Hugging Face's diffusers UNet models
"""


class HFNet(torch.nn.Module):
    def __init__(self, block_channels=[64, 128, 256],
                 channels=1,
                 cond_channels=1,
                 norm_num_groups=32,
                 dropout=0.0,
                 attn_up_and_down=False):
        """
        Parameters
        ----------
        block_channels : list[int]
            The number of channels for each block.
            The length of the list determines the number of blocks.
        channels : int
            The number of input (and output) channels in the model
        """
        super().__init__()
        num_blocks = len(block_channels)
        if attn_up_and_down:
            down_block_types = (['DownBlock2D'] +
                                ['AttnDownBlock2D']*(num_blocks-1))
            up_block_types = (['AttnUpBlock2D']*(num_blocks-1) +
                              ['UpBlock2D'])
        else:
            down_block_types = ['DownBlock2D']*num_blocks
            up_block_types = ['UpBlock2D']*num_blocks
        in_channels = channels + cond_channels
        self.model = diffusers.UNet2DModel(
                        in_channels=in_channels,
                        out_channels=channels,
                        block_out_channels=block_channels,
                        down_block_types=down_block_types,
                        up_block_types=up_block_types,
                        norm_num_groups=norm_num_groups,
                        dropout=dropout)

    def forward(self, x, t):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, channels, H, W)
        t : torch.Tensor of shape (B,)

        Returns
        -------
        torch.Tensor of shape (B, channels, H, W)
        """
        return self.model(x, t).sample


class HFNetUncond(HFNet):
    def __init__(self, block_channels=[64, 128, 256],
                 channels=1,
                 norm_num_groups=32,
                 dropout=0.0,
                 attn_up_and_down=False):
        super().__init__(block_channels,
                         channels,
                         cond_channels=0,
                         norm_num_groups=norm_num_groups,
                         dropout=dropout,
                         attn_up_and_down=attn_up_and_down)


class HFNetCond(HFNet):
    def __init__(self, block_channels=[64, 128, 256],
                 channels=1,
                 cond_channels=1,
                 norm_num_groups=32,
                 dropout=0.0,
                 attn_up_and_down=False):
        super().__init__(block_channels,
                         channels,
                         cond_channels=cond_channels,
                         norm_num_groups=norm_num_groups,
                         dropout=dropout,
                         attn_up_and_down=attn_up_and_down)

    def forward(self, x, t, y):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, channels, H, W)
        t : torch.Tensor of shape (B,)
        y : torch.Tensor of shape (B, conditional_channels, H, W)

        Returns
        -------
        torch.Tensor of shape (B, channels, H, W)
        """
        x = torch.concatenate([x, y], axis=1)  # (nbatch, 2, w, h)
        return super().forward(x, t)
