import copy

import torch
import tqdm
import tqdm.notebook
import matplotlib.pyplot as plt
import safetensors.torch


class SDETrainer(object):
    def __init__(self, model, scheduler, train_dataloader,
                 test_dataloader=None,
                 conditional=True,
                 loss_type="mse",
                 loss_scale_factor=1.0,
                 device="cpu"):

        """
        Class to train a model accoding to the framework in "Score-Based
        Generative Modeling Through Stochastic Differential Equations" by Song
        et al.

        Parameters
        ----------
        model : torch.nn.Module taking as input:
                x : torch.Tensor of shape (B, C, H, W), the original noise
                t : torch.Tensor of shape (B,)
                and, if conditional=True
                y : torch.Tensor of shape (B, ...), the conditional data
                and as output
                torch.Tensor of shape (B, C, H, W)
        scheduler : SDEScheduler
        train_dataloader : torch.utils.data.DataLoader
            train_dataloader spitting out at each iteration a (x, y) pair
        test_dataloader : torch.utils.data.DataLoader | None
            test_dataloader for validation. If None, no validation is
            performed.
        conditional : bool
            whether we are dealing with a conditional or unconditional model
        loss_type : str
            what kind of loss are we using. Options: ["mse", "huber"].
        loss_scale_factor : float
        device : torch.device
        """

        self.model = model
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.conditional = conditional
        self.loss_type = loss_type
        self.loss_scale_factor = loss_scale_factor
        self.device = device
        self.train_loss_history = []
        self.test_loss_history = []
        self.best_test_model = copy.deepcopy(self.model)
        self.best_test_loss = float("inf")
        self.possible_losses = ["huber", "mse"]
        self.set_loss(self.loss_type)

    def set_loss(self, loss_type):
        """
        Set the loss function to be used.

        Parameters
        ----------
        loss_type : str
            what kind of loss are we using. Options: ["mse", "huber"].
        """
        assert (loss_type in self.possible_losses)
        if loss_type == "mse":
            self.loss_metric = torch.nn.MSELoss(reduction="none")
        elif loss_type == "huber":
            self.loss_metric = torch.nn.HuberLoss(reduction="none")

    def set_optimizer_and_scheduler(self,
                                    optimizer=None,
                                    step_scheduler=None,
                                    epoch_scheduler=None):

        """
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
        step_scheduler : torch.optim.schedulers
        epoch_scheduler : torch.optim.schedulers
        """

        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer

        if step_scheduler is None:
            self.step_scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer,
                    len(self.train_dataloader)
                )
            )

        else:
            self.step_scheduler = step_scheduler

        if epoch_scheduler is None:
            self.epoch_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: max(0.05, 0.8 ** epoch)
            )

        else:
            self.epoch_scheduler = epoch_scheduler

    def loss_fn(self, x, t, y=None):

        """
        Parameters
        ---------
        x : torch.Tensor of shape (B, C, H, W), the original noise
        t : torch.Tensor of shape (B,)
        y : None or torch.Tensor of shape (B, ...), the conditional data,
        depending on whether we are dealing with a conditional or
        unconditional model
        """

        nbatch = x.shape[0]
        if self.conditional:
            assert (y is not None)
        elif not self.conditional:
            assert (y is None)
        mean = self.scheduler.mean(t, x)  # (nbatch, C, H, W)
        std = self.scheduler.std(t)  # (nbatch,)
        std_shape = [nbatch] + [1]*(len(mean.shape)-1)
        std = std.reshape(std_shape)  # (nbatch, 1, 1, 1)
        noise = torch.randn_like(x)  # (nbatch, C, H, W)
        # std = std[:, None, None, None] #(nbatch, 1, 1, 1)
        x_noised = mean + std*noise  # (nbatch, C, H, W)
        if self.conditional:
            score = self.model(x_noised, t, y)  # (nbatch, C, H, W)
        elif not self.conditional:
            score = self.model(x_noised, t)  # (nbatch, C, H, W)

        # loss = ((std*score - noise)**2).mean(dim=0).sum()
        # loss = ((std*score+noise)**2).mean()
        # loss = (score + noise/std).mean()
        # loss = self.loss_metric(std*score, -noise).mean()
        loss = (1/std*self.loss_metric(std*score, -noise)).mean()
        return self.loss_scale_factor*loss

    def train(self, nepochs=20, on_notebook=False):

        """
        Train our model

        Parameters
        ----------
        nepochs : int
            The number of epochs to train for
        """

        assert (hasattr(self, "optimizer"))
        # tqdm_epoch is an iterator over epochs
        tqdm_epoch = (tqdm.trange(nepochs) if not on_notebook
                      else tqdm.notebook.trange(nepochs))
        for epoch in tqdm_epoch:
            # Training loop
            self.model.train()  # Set model to train mode

            # Average loss over the epoch. We accumulate then divide.
            avg_train_loss = 0.

            # Number of training items in the epoch. We accumulate then divide.
            num_items = 0

            # tqdm_step is an iterator over batches
            tqdm_step = (tqdm.tqdm(self.train_dataloader) if not on_notebook
                         else tqdm.notebook.tqdm(self.train_dataloader))

            for data in tqdm_step:
                self.optimizer.zero_grad()  # Reset gradients
                if self.conditional:
                    x, y = data  # Unpack data
                    x = x.to(self.device)  # Move to device
                    y = y.to(self.device)  # Move to device
                elif not self.conditional:
                    x = data  # Only unpack x
                    x = x.to(self.device)  # Move to device
                    y = None  # No y
                t = self.scheduler.sample_time(x.shape[0])  # Sample t
                t = t.to(self.device)  # Move to device
                loss = self.loss_fn(x, t, y)  # Compute loss
                loss.backward()  # Backpropagate
                self.optimizer.step()  # Update parameters
                self.step_scheduler.step()  # Update learning rate
                avg_train_loss += loss.item()*x.shape[0]  # Accumulate loss
                num_items += x.shape[0]  # Accumulate number of items
            avg_train_loss = avg_train_loss/num_items  # Average loss
            self.train_loss_history.append(avg_train_loss)  # Append to history
            self.epoch_scheduler.step()  # Update learning rate

            # In case of only having training loop
            if self.test_dataloader is None:
                description = 'Avg Train Loss: {:5f}'.format(avg_train_loss)
                tqdm_epoch.set_description(description)
                continue  # We break our loop here, no need for evaluation loop

            # Evaluation loop
            with torch.no_grad():
                self.model.eval()  # Set model to eval mode

                # Average loss over the epoch. We accumulate then divide.
                avg_test_loss = 0.0

                # Number of training items in the epoch.
                # We accumulate then divide.
                num_items = 0

                tqdm_step = (tqdm.tqdm(self.test_dataloader) if not on_notebook
                             else tqdm.notebook.tqdm(self.test_dataloader))
                for data in tqdm_step:
                    if self.conditional:
                        x, y = data  # Unpack data
                        x = x.to(self.device)  # Move to device
                        y = y.to(self.device)  # Move to device
                    elif not self.conditional:
                        x = data  # Only unpack x
                        x = x.to(self.device)  # Move to device
                        y = None  # No y
                    t = self.scheduler.sample_time(x.shape[0])  # Sample t
                    t = t.to(self.device)  # Move to device
                    loss = self.loss_fn(x, t, y)  # Compute loss
                    avg_test_loss += loss.item()*x.shape[0]  # Accumulate loss
                    num_items += x.shape[0]  # Accumulate number of items
                avg_test_loss = avg_test_loss/num_items   # Average loss

                # Append to history
                self.test_loss_history.append(avg_test_loss)

                if avg_test_loss < self.best_test_loss:  # If best test loss

                    # Update best test loss
                    self.best_test_loss = avg_test_loss

                    # Update best model
                    self.best_test_model = copy.deepcopy(self.model)

            # Display loop state
            description = (
                f'Avg Train Loss: {avg_train_loss:5f}, ' +
                f'Avg Test Loss: {avg_test_loss:.5f}'
            )

            tqdm_epoch.set_description(description)

    def save_model(self, path, mode="best"):

        """
        Parameters
        ----------
        path : str
            path to save model
        mode : ["best", "current"]
            whether to save best model (in evaluation) or current model
        """

        model = self.best_test_model if mode == "best" else self.model
        safetensors.torch.save_model(model, path)

    def plot_history(self, logy=False):

        """
        Plot the training history

        Parameters
        ----------
        logy : bool
            whether to plot the y axis in log scale
        """

        train_epochs = list(range(1, len(self.train_loss_history)+1))
        plt.plot(train_epochs, self.train_loss_history, 'bo', label='train')
        test_epochs = list(range(1, len(self.test_loss_history)+1))
        plt.plot(test_epochs, self.test_loss_history, 'ro', label='test')
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Training history")
        plt.legend()
        if logy:
            plt.semilogy()
