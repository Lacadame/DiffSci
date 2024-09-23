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
    model = diffsci.models.PUNetUncond(net_channels)
    moduleconfig = diffsci.models.KarrasModuleConfig.from_edm()
    module = diffsci.models.KarrasModule(model, moduleconfig)
    trainer = lightning.Trainer(max_epochs=max_epochs,
                                enable_checkpointing=False,
                                logger=False,
                                fast_dev_run=True)
    trainer.fit(model=module, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    images = module.sample(nsamples=4,
                           shape=[1, 28, 28],
                           nsteps=10)
    assert tuple(images.shape) == (4, 1, 28, 28)
    images = module.sample(nsamples=4,
                           shape=[1, 28, 28],
                           nsteps=10,
                           maximum_batch_size=2,
                           record_history=True)
    assert tuple(images.shape) == (11, 4, 1, 28, 28)
    mask = torch.ones(4, 1, 28, 28)
    mask[:, :, 10:, :] = 0
    x_orig = images[0]
    x_inpainted = module.inpaint(x_orig,
                                 mask,
                                 nsteps=10,
                                 record_history=False)
    assert tuple(x_inpainted.shape) == (4, 1, 28, 28)
    x_inpainted = module.inpaint(x_orig,
                                 mask,
                                 nsteps=30,
                                 record_history=True)
    assert tuple(x_inpainted.shape) == (31, 4, 1, 28, 28)
    x_repainted = module.repaint(x_orig,
                                 mask,
                                 nsteps=20,
                                 record_history=False)
    # print(x_repainted.shape)
    assert tuple(x_repainted.shape) == (4, 1, 28, 28)
    x_inpainted = module.repaint(x_orig,
                                 mask,
                                 nsteps=50,
                                 record_history=True)
    print(x_inpainted.shape)
    # assert tuple(x_inpainted.shape) == (11, 4, 1, 28, 28)


def test_mnist_lightning_cond(batch_size=2,
                              net_channels=4,
                              max_epochs=1):
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, mnist_dataset):
            self.mnist_dataset = mnist_dataset

        def __len__(self):
            return len(self.mnist_dataset)

        def __getitem__(self, idx):
            x, y = self.mnist_dataset[idx]
            return x, y

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

    class ConditionalEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(10, net_channels)

        def forward(self, y):
            return self.embedding(y)

    conditional_embedding = ConditionalEmbedding()
    modelconfig = diffsci.models.PUNetGConfig(model_channels=net_channels)
    model = diffsci.models.PUNetG(modelconfig,
                                  conditional_embedding=conditional_embedding)
    moduleconfig = diffsci.models.KarrasModuleConfig.from_edm()
    module = diffsci.models.KarrasModule(model, moduleconfig,
                                         conditional=True)
    trainer = lightning.Trainer(max_epochs=max_epochs,
                                enable_checkpointing=False,
                                logger=False,
                                fast_dev_run=True)
    trainer.fit(model=module, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    _, y = train_dataset[0]
    images = module.sample(nsamples=4,
                           shape=[1, 28, 28],
                           nsteps=10,
                           y=torch.tensor(y, dtype=torch.long))
    assert tuple(images.shape) == (4, 1, 28, 28)
    # images = module.sample(nsamples=4,
    #                        shape=[1, 28, 28],
    #                        nsteps=10,
    #                        maximum_batch_size=2,
    #                        record_history=True)
    # assert tuple(images.shape) == (11, 4, 1, 28, 28)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
                        default="cond",
                        choices=["cond", "uncond"],
                        help="Task to run")
    parser.add_argument("--batch_size",
                        default=5,
                        type=int,
                        help="Batch size")
    parser.add_argument("--net_channels",
                        default=4,
                        type=int,
                        help="Number of channels in the network")
    parser.add_argument("--max_epochs",
                        default=1,
                        type=int,
                        help="Number of epochs to train for")
    args = parser.parse_args()
    task = args.task

    test_mnist_lightning_uncond(batch_size=args.batch_size,
                                net_channels=args.net_channels,
                                max_epochs=args.max_epochs)
    test_mnist_lightning_cond(batch_size=args.batch_size,
                              net_channels=args.net_channels,
                              max_epochs=args.max_epochs)
    print("All tests passed")
