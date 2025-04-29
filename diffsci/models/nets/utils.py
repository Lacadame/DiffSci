import math

import torch


class GaussianFourierProjection(torch.nn.Module):
    def __init__(self, embed_dim, scale=30.0):

        """
        Parameters
        ----------
        embed_dim : int
            The dimension of the embedding
        scale : float
            The scale of the gaussian distribution
        """

        super().__init__()
        self.W = torch.nn.Parameter(
            torch.randn(embed_dim//2)*scale,
            requires_grad=False
        )

    def forward(self, x):

        """
        Parameters
        ----------
        x : torch.Tensor of shape (..., 1)

        Returns
        -------
        x_proj : torch.Tensor of shape (..., embed_dim)
        """

        x = x[..., None]  # (..., 1)
        x_proj = 2*math.pi*x*self.W  # (..., embed_dim//2)
        x_proj = torch.cat(
            [torch.sin(x_proj), torch.cos(x_proj)], dim=-1
        )  # (..., embed_dim)
        return x_proj


class GroupRMSNorm(torch.nn.Module):
    def __init__(self,
                 num_groups,
                 num_channels,
                 eps=1e-5,
                 affine=True):
        """
        Group RMS normalization layer. This layer divides the channels into
        groups and normalizes along the (C//G, *) dimensions.

        Parameters
        ----------
        num_groups : int
            The number of groups to divide the channels
        num_channels : int
            The number of channels expected in the input
        eps : float
            The epsilon value to avoid division by zero
        affine : bool
            Whether to apply an affine transformation to the input
        """
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = torch.nn.Parameter(torch.ones(num_channels))
            self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, C, *)

        Returns
        -------
        x : torch.Tensor of shape (B, C, *)
        """
        B, C = x.shape[:2]
        G = self.num_groups
        x = x.view(B, G, C//G, *x.shape[2:])
        normalize_dims = tuple(range(2, x.dim()))

        x = x / (torch.sqrt(x.pow(2).mean(dim=normalize_dims, keepdim=True)
                            + self.eps))
        x = x.view(B, C, *x.shape[3:])
        if self.affine:
            w = self.weight.view(1, C, *([1]*(x.dim()-2)))
            b = self.bias.view(1, C, *([1]*(x.dim()-2)))
            x = x*w + b
        return x
