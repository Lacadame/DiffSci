import torch


class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dim=2, num_groups=8):
        super().__init__()
        self.dim = dim
        self.conv1 = self.get_conv_cls()(channels, channels, kernel_size, padding='same')
        self.conv2 = self.get_conv_cls()(channels, channels, kernel_size, padding='same')
        self.norm1 = torch.nn.GroupNorm(num_groups, channels)
        self.norm2 = torch.nn.GroupNorm(num_groups, channels)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = torch.nn.functional.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = torch.nn.functional.silu(h)
        h = self.conv2(h)
        return x + h

    def get_conv_cls(self):
        if self.dim == 1:
            return torch.nn.Conv1d
        elif self.dim == 2:
            return torch.nn.Conv2d
        elif self.dim == 3:
            return torch.nn.Conv3d
        else:
            raise ValueError(f"Invalid dimension: {self.dim}")


class MinimalResNet(torch.nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_classes=1,
        model_channels=32,
        n_layers=8,
        dim=2,
        kernel_size=3,
        num_groups=8,
    ):
        super().__init__()
        self.dim = dim
        self.in_conv = self.get_conv_cls()(in_channels, model_channels, kernel_size, padding='same')

        self.res_blocks = torch.nn.ModuleList([
            ResBlock(model_channels, kernel_size=kernel_size, dim=dim, num_groups=num_groups)
            for _ in range(n_layers)
        ])

        self.pool = self.get_pool_cls()(1)
        self.out = torch.nn.Linear(model_channels, out_classes)

    def get_conv_cls(self):
        if self.dim == 1:
            return torch.nn.Conv1d
        elif self.dim == 2:
            return torch.nn.Conv2d
        elif self.dim == 3:
            return torch.nn.Conv3d
        else:
            raise ValueError(f"Invalid dimension: {self.dim}")

    def get_pool_cls(self):
        if self.dim == 1:
            return torch.nn.AdaptiveAvgPool1d
        elif self.dim == 2:
            return torch.nn.AdaptiveAvgPool2d
        elif self.dim == 3:
            return torch.nn.AdaptiveAvgPool3d
        else:
            raise ValueError(f"Invalid dimension: {self.dim}")

    def forward(self, x):
        h = self.in_conv(x)

        for block in self.res_blocks:
            h = block(h)

        h = self.pool(h)
        h = h.view(h.size(0), -1)
        return self.out(h)
