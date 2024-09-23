import math

import torch


class MagnitudePreservingLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features,
                                                     in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        self.epsilon = 1e-4

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))
        fan_in = self.weight[0].numel()
        w = normalize(self.weight)/math.sqrt(fan_in)
        return torch.nn.functional.linear(x, w, self.bias)


class MagnitudePreservingConv2d(torch.nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_channels,
                                                     in_channels,
                                                     kernel_size,
                                                     kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.epsilon = 1e-4

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))
        fan_in = self.weight[0].numel()
        w = normalize(self.weight)/math.sqrt(fan_in)
        return torch.nn.functional.conv2d(x,
                                          w,
                                          self.bias,
                                          self.stride,
                                          self.padding)


class MagnitudePreservingConv3d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(out_channels,
                        in_channels,
                        kernel_size,
                        kernel_size,
                        kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.epsilon = 1e-4

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))
        fan_in = self.weight[0].numel()
        w = normalize(self.weight)/math.sqrt(fan_in)
        return torch.nn.functional.conv3d(x,
                                          w,
                                          self.bias,
                                          self.stride,
                                          self.padding)


def normalize(x, eps=1e-4):
    dim = list(range(1, x.ndim))
    n = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    alpha = math.sqrt(n.numel()/x.numel())
    return x / torch.add(eps, n, alpha=alpha)
