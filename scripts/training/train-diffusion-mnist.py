import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import lightning
import lightning.pytorch.callbacks as pl_callbacks
from torch.utils.data import Dataset, Subset
import torchvision
import torchvision.transforms as transforms

import diffsci.models


def drop_label_collate(batch):
    """
    Standard MNIST returns (image, label). 
    This function discards the label and returns just the stacked images.
    """
    images = [item[0] for item in batch]
    return torch.stack(images)


def main():

    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    # Set parameters
    device_id = 7
    batch_size = 32
    model_channels = 128
    n_epochs = 500
    every_n_epochs = 100
    learning_rate = 1e-4
    n_tsamples = 20  # Number of training samples to use; set to None to use the full dataset
    checkpoint_dir = f"/home/ubuntu/repos/DiffSci/savedmodels/experimental/20260428-bps-mnist-128ch-20samples"

    # Load MNIST datasets
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(
        root='/home/ubuntu/repos/DiffSci/saveddata/external', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.MNIST(
        root='/home/ubuntu/repos/DiffSci/saveddata/external', train=False, download=True, transform=transform)

    # Restrict the training set to a fixed subset of n_tsamples images.
    # The indices are drawn deterministically (numpy seed set above) so the same
    # subset is used across runs and the model only ever sees these samples.
    if n_tsamples is not None:
        if n_tsamples > len(train_dataset):
            raise ValueError(
                f"n_tsamples ({n_tsamples}) exceeds dataset size ({len(train_dataset)})"
            )
        train_indices = np.random.permutation(len(train_dataset))[:n_tsamples]
        train_dataset = Subset(train_dataset, train_indices.tolist())

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
    modelconfig = diffsci.models.PUNetGConfig(model_channels=model_channels)
    model = diffsci.models.PUNetG(modelconfig)
    moduleconfig = diffsci.models.KarrasModuleConfig.from_edm()
    module = diffsci.models.KarrasModule(model, moduleconfig, conditional=False)

    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate)
    module.set_optimizer_and_scheduler(optimizer) 
    
    callbacks = [diffsci.models.callbacks.NanToZeroGradCallback()]

    # Create directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = pl_callbacks.ModelCheckpoint(
        dirpath=os.path.join(checkpoint_dir, 'checkpoints'),
        filename='model-{epoch:03d}-{val_loss:.6f}',
        every_n_epochs=every_n_epochs,
        save_top_k=-1,
        monitor='val_loss',
        mode='min',
        save_last=False
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
    print(f"Training on {len(train_dataset)} samples.")

    trainer.fit(module, train_dataloader, val_dataloader)

    # save configs in a txt file
    with open(os.path.join(checkpoint_dir, 'configs.txt'), 'w') as f:
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Model channels: {model_channels}\n")
        f.write(f"Number of epochs: {n_epochs}\n")
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Number of training samples: {n_tsamples}\n")
        f.write(f"Preconditioning: edm \n")
        f.write(f"Optimizer: AdamW \n")

if __name__ == "__main__":
    main()