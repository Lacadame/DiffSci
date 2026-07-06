import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import lightning
import lightning.pytorch.callbacks as pl_callbacks
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

import diffsci.models


def drop_label_collate(batch):
    """
    Standard CIFAR-10 returns (image, label).
    This function discards the label and returns just the stacked images.
    """
    images = [item[0] for item in batch]
    return torch.stack(images)


def main():

    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Set parameters
    device_id = 5
    batch_size = 32
    model_channels = 128
    n_epochs = 100
    learning_rate = 1e-4
    checkpoint_dir = f"/home/ubuntu/repos/DiffSci/savedmodels/experimental/20260304-bps-cifar10-128ch-vp2"

    # Load CIFAR-10 datasets
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(
        root='/home/ubuntu/repos/DiffSci/saveddata/external', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.CIFAR10(
        root='/home/ubuntu/repos/DiffSci/saveddata/external', train=False, download=True, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=drop_label_collate
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=drop_label_collate
    )


    # Define model
    modelconfig = diffsci.models.PUNetGConfig(
        model_channels=model_channels,
        input_channels=3,
        output_channels=3
    )
    model = diffsci.models.PUNetG(modelconfig)
    moduleconfig = diffsci.models.KarrasModuleConfig.from_vp()
    module = diffsci.models.KarrasModule(model, moduleconfig, conditional=False)

    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate)
    module.set_optimizer_and_scheduler(optimizer)

    callbacks = [diffsci.models.callbacks.NanToZeroGradCallback()]

    # Create directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath=os.path.join(checkpoint_dir, 'checkpoints'),
        filename='model-{epoch:03d}-{val_loss:.6f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True
    )

    trainer = lightning.Trainer(
        accelerator="gpu",
        devices=[device_id],
        max_epochs=n_epochs,
        default_root_dir=checkpoint_dir,
        gradient_clip_val=0.5,
        callbacks=callbacks + [checkpoint_callback],
        enable_checkpointing=True
    )

    print(f"Starting training on device {device_id}, with batch size {batch_size} and learning rate {learning_rate} for {n_epochs} epochs.")

    trainer.fit(module, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
