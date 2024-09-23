import os
import pathlib
import argparse

import torch
import torchvision
import lightning

import diffsci.models


SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = pathlib.Path(os.path.dirname(SCRIPT_PATH))
MAINFOLDER = SCRIPT_DIR.parent
DATAPATH = MAINFOLDER/"saveddata"
MODELSPATH = MAINFOLDER/"savedmodels"


def test_mnist_lightning_cond(batch_size=16,
                              net_channels=16,
                              max_epochs=20):
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, mnist_dataset, scale_factor=4):
            self.mnist_dataset = mnist_dataset
            self.scale_factor = scale_factor

        def __len__(self):
            return len(self.mnist_dataset)

        def __getitem__(self, idx):
            x, _ = self.mnist_dataset[idx]
            y = self.destroy_image(x, self.scale_factor)
            return x, y

        def destroy_image(self, x, scale_factor=2):
            # x : batch_size, 1, 8, 8
            squeeze_later = False
            if len(x.shape) == 3:
                squeeze_later = True
                x = x.unsqueeze(0)
            down_fn = torch.nn.AvgPool2d(scale_factor)
            up_fn = torch.nn.Upsample(scale_factor=scale_factor)
            y = up_fn(down_fn(x))
            if squeeze_later:
                y = y.squeeze(0)
            return y

    mnist_dataset = torchvision.datasets.MNIST(
                        DATAPATH/'external',
                        train=True,
                        transform=torchvision.transforms.ToTensor())
    dataset = CustomDataset(mnist_dataset)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
                                    dataset,
                                    [train_size,
                                     val_size])

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=4)
    model = diffsci.models.PUNetCond(net_channels)
    scheduler = diffsci.models.ddpmv1.DDPMScheduler()
    module = diffsci.models.ddpmv1.DDPMModule(model,
                                              scheduler,
                                              conditional=True)
    trainer = lightning.Trainer(max_epochs=max_epochs,
                                enable_checkpointing=False,
                                logger=False,
                                fast_dev_run=True)
    trainer.fit(model=module, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)


def test_mnist_lightning_uncond(batch_size=16,
                                net_channels=16,
                                max_epochs=20):
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, mnist_dataset):
            self.mnist_dataset = mnist_dataset

        def __len__(self):
            return len(self.mnist_dataset)

        def __getitem__(self, idx):
            x, _ = self.mnist_dataset[idx]
            return x

    mnist_dataset = torchvision.datasets.MNIST(
                        DATAPATH/'external',
                        train=True,
                        transform=torchvision.transforms.ToTensor())
    dataset = CustomDataset(mnist_dataset)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
                                    dataset,
                                    [train_size,
                                     val_size])

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=4)
    model = diffsci.models.AkshahyNetUncond(net_channels)
    scheduler = diffsci.models.DDPMScheduler()
    module = diffsci.models.DDPMModule(model, scheduler,
                                       conditional=False,
                                       loss_type="huber")
    trainer = lightning.Trainer(max_epochs=max_epochs,
                                enable_checkpointing=False,
                                logger=False)
    trainer.fit(model=module, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
                        default="cond",
                        choices=["cond", "uncond"],
                        help="Task to run")
    parser.add_argument("--batch_size",
                        default=256,
                        type=int,
                        help="Batch size")
    parser.add_argument("--net_channels",
                        default=16,
                        type=int,
                        help="Number of channels in the network")
    parser.add_argument("--max_epochs",
                        default=1,
                        type=int,
                        help="Number of epochs to train for")
    args = parser.parse_args()
    task = args.task
    if task == "cond":
        test_mnist_lightning_cond(batch_size=args.batch_size,
                                  net_channels=args.net_channels,
                                  max_epochs=args.max_epochs)
    elif task == "uncond":
        test_mnist_lightning_uncond(batch_size=args.batch_size,
                                    net_channels=args.net_channels,
                                    max_epochs=args.max_epochs)
    print("All tests passed")
