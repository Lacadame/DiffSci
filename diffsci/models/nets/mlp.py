import torch


class MLPUncond(torch.nn.Module):
    def __init__(self, dim, hidden_dims=[10],
                 nonlinearity=torch.nn.ReLU()):

        """
        A MLP with variable number of hidden layers, with no conditioning on y.

        Parameters
        ----------
        dim : int
            Input dimension.
        hidden_dims : list of int
            A list containing dimensions of each hidden layer.
        nonlinearity : torch.nn.Module, optional
            The non-linear activation function to use.
            Defaults to torch.nn.ReLU().
        """

        super().__init__()
        self.dim = dim
        layers = []  # Initialize an empty list to hold layers
        in_dim = dim + 1  # The input dimension includes 't'
        # Loop through each hidden layer and add it to the list
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(nonlinearity)
            in_dim = hidden_dim
        # Add the final output layer and convert to a sequential model
        layers.append(torch.nn.Linear(in_dim, dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x, t):

        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor of shape [nbatch, dim]
            Input tensor
        t : torch.Tensor of shape [nbatch]
            Time tensor

        Returns
        -------
        torch.Tensor of shape [nbatch, dim]
            Output tensor.
        """

        t = t[..., None]
        x_ = torch.cat([x, t], dim=-1)  # [nbatch, dim + 1 + ydim]
        return self.net(x_)


class MLPCond(torch.nn.Module):

    def __init__(self, dim, ydim, hidden_dims=[10],
                 nonlinearity=torch.nn.ReLU()):

        """
        A MLP with variable number of hidden layers, with conditioning on y.

        Parameters
        ----------
        dim : int
            Input dimension.
        ydim : int
            Input dimension of the conditioning vector.
        hidden_dims : list of int
            A list containing dimensions of each hidden layer.
        nonlinearity : torch.nn.Module, optional
            The non-linear activation function to use.
            Defaults to torch.nn.ReLU().
        """

        super().__init__()
        self.dim = dim
        self.ydim = ydim
        layers = []  # Initialize an empty list to hold layers
        in_dim = dim + 1 + ydim  # The input dimension includes 't'
        # Loop through each hidden layer and add it to the list
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(nonlinearity)
            in_dim = hidden_dim
        # Add the final output layer and convert to a sequential model
        layers.append(torch.nn.Linear(in_dim, dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x, t, y):

        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor of shape [nbatch, dim]
            Input tensor
        t : torch.Tensor of shape [nbatch]
            Time tensor
        y : torch.Tensor of shape [nbatch, ydim]
            Conditional input tensor

        Returns
        -------
        torch.Tensor of shape [nbatch, dim]
            Output tensor.
        """

        t = t[..., None]
        x_ = torch.cat([x, t, y], dim=-1)  # [nbatch, dim + 1 + ydim]
        return self.net(x_)
