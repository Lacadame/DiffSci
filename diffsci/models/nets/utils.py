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


class DateGaussianFourierProjection(torch.nn.Module):
    """Compute the Gaussian Fourier Projection for the day and month"""
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
        # Initialize weights for days and months separately
        self.W_day = torch.nn.Parameter(
            torch.randn(embed_dim // 2) * scale,
            requires_grad=False
        )
        self.W_month = torch.nn.Parameter(
            torch.randn(embed_dim // 2) * scale,
            requires_grad=False
        )

    def forward(self, dates):
        """
        Assume dates is a tensor of shape (Batch, 2) with the format
        [[month,day]] Split the dates into day and month components

        Parameters
        ----------
        dates : torch.Tensor of shape (..., 2)

        Returns
        -------
        combined_features : torch.Tensor of shape (..., embed_dim)
        """

        day = dates[:, 1] / 31 * 2 * math.pi
        month = dates[:, 0] / 12 * 2 * math.pi

        # Project days
        day_proj = day.unsqueeze(-1) * self.W_day
        day_features = torch.cat([torch.sin(day_proj),
                                  torch.cos(day_proj)], dim=-1)

        # Project months
        month_proj = month.unsqueeze(-1) * self.W_month
        month_features = torch.cat([torch.sin(month_proj),
                                    torch.cos(month_proj)], dim=-1)

        # Combine day and month features
        combined_features = day_features + month_features
        # Element-wise sum or another combination method

        return combined_features  # [batch, embed_dim]


class GeoGaussianFourierProjection(torch.nn.Module):
    """Compute the Gaussian Fourier Projection for latitude and longitude."""
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
        # Initialize weights for latitude and longitude separately
        self.W_lat = torch.nn.Parameter(
            torch.randn(embed_dim // 2) * scale,
            requires_grad=False
        )
        self.W_long = torch.nn.Parameter(
            torch.randn(embed_dim // 2) * scale,
            requires_grad=False
        )

    def forward(self, coordinates):
        """
        Assume coordinates is a tensor of shape (Batch, 2) with the format
        [[latitude, longitude]]. Split the coordinates into latitude and 
        longitude components.

        Parameters
        ----------
        coordinates : torch.Tensor of shape (..., 2)

        Returns
        -------
        combined_features : torch.Tensor of shape (..., embed_dim)
        """
        # Normalize and scale latitudes and longitudes to radians
        lat = (coordinates[:, 0] + 90) / 180 * math.pi
        # Convert [-90, 90] to [0, pi]
        long = (coordinates[:, 1] + 180) / 360 * 2 * math.pi
        # Convert [-180, 180] to [0, 2*pi]

        # Project latitudes
        lat_proj = lat.unsqueeze(-1) * self.W_lat
        lat_features = torch.cat([torch.sin(lat_proj),
                                  torch.cos(lat_proj)], dim=-1)

        # Project longitudes
        long_proj = long.unsqueeze(-1) * self.W_long
        long_features = torch.cat([torch.sin(long_proj),
                                   torch.cos(long_proj)], dim=-1)

        # Combine latitude and longitude features
        combined_features = lat_features + long_features
        # Element-wise sum or another combination method

        return combined_features  # [batch, embed_dim]


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
