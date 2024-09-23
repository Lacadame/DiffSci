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
        x = x*torch.sqrt(self.running_var + self.eps) + self.running_mean
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
