import torch


class EDMBatchNorm(torch.nn.Module):
    def __init__(self,
                 sigma=1.0,
                 eps=1e-5,
                 momentum=0.01,
                 ):
        super().__init__()
        self.sigma = sigma
        self.momentum = momentum
        self.use_running_mean = False
        self.register_buffer("eps", torch.ones([])*eps)
        self.reset_parameters()

    def reset_parameters(self):
        self.register_buffer('_running_mean', torch.zeros([]))
        self.register_buffer('_running_var', torch.ones([]))
        self.has_running_mean = False
        self.has_running_var = False

    def forward(self, input):
        if not self.use_running_mean:
            batch_mean = input.mean()
            batch_var = input.var(unbiased=False)
            with torch.no_grad():
                if not self.has_running_mean:
                    self.running_mean = batch_mean
                    self.running_var = batch_var
                else:
                    self.running_mean = (
                        (1 - self.momentum) * self.running_mean +
                        self.momentum * batch_mean)
                    self.running_var = (
                        (1 - self.momentum) * self.running_var +
                        self.momentum * batch_var)
            x = (input - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            x = ((input - self.running_mean) /
                 torch.sqrt(self.running_var + self.eps))

        # Apply custom standard deviation
        x = x * self.sigma

        return x

    def normalize(self, input):  # Alias for forward
        return self(input)

    def unnormalize(self, input):
        x = input/self.sigma

        # QUICK FIX: Just for inference running mean not depending on
        # self.use_running_mean flag. Should be fixed in the future.
        x = x*torch.sqrt(self._running_var + self.eps) + self._running_mean
        return x

    @property
    def running_mean(self):
        if not self.has_running_mean:
            return 0.0
        else:
            return self._running_mean

    @property
    def running_var(self):
        if not self.has_running_var:
            return 1.0
        else:
            return self._running_var

    @running_mean.setter
    def running_mean(self, value):
        if not self.has_running_mean:
            self.has_running_mean = True
        self._running_mean = value

    @running_var.setter
    def running_var(self, value):
        if not self.has_running_var:
            self.register_buffer("_running_var", value)
        self._running_var = value


class DimensionAgnosticBatchNorm(torch.nn.Module):
    def __init__(self,
                 num_channels: int | None = None,
                 eps: float = 1e-5,
                 affine: bool = False,
                 momentum: float = 0.1,
                 sigma: float = 1.0):
        super().__init__()
        if num_channels is None:
            self.num_channels = None
        else:
            self.num_channels = num_channels
        self.nc = self.num_channels if self.num_channels is not None else 1  # In None case, 1 will be broadcasted anyway
        self.eps = eps
        self.affine = affine
        self.momentum = momentum
        self.sigma = sigma

        if affine:
            self.weight = torch.nn.Parameter(torch.ones(self.nc))
            self.bias = torch.nn.Parameter(torch.zeros(self.nc))

        self.register_buffer('running_mean', torch.zeros(self.nc))
        self.register_buffer('running_var', torch.ones(self.nc))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        For backward compatibility with old code.
        """
        return self(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (N, C, *spatial_dims)
        dims = list(range(len(x.shape)))
        # Remove channel dim from reduction
        dims.pop(1)

        if self.training:
            mean = x.mean(dim=dims)
            var = x.var(dim=dims, unbiased=False)

            # Update running stats
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.detach()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.detach()
        else:
            mean = self.running_mean
            var = self.running_var

        # Reshape for broadcasting
        shape = [1, self.nc] + [1] * (len(x.shape) - 2)
        mean = mean.view(shape)
        var = var.view(shape)

        x = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            weight = self.weight.view(shape)
            bias = self.bias.view(shape)
            x = x * weight + bias

        x = x * self.sigma

        return x

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        For backward compatibility with old code.
        """
        return self.unnorm(x)

    def unnorm(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape for broadcasting
        shape = [1, self.nc] + [1] * (len(x.shape) - 2)

        x = x / self.sigma

        if self.affine:
            weight = self.weight.view(shape)
            bias = self.bias.view(shape)
            x = (x - bias) / weight

        mean = self.running_mean.view(shape)
        var = self.running_var.view(shape)

        x = x * torch.sqrt(var + self.eps) + mean
        return x


class IdentityBatchNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def unnorm(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x
