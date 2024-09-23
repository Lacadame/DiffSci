import pathlib

import torch
import torchvision

import diffsci.models
from diffsci.models.nets import punet


class TestPUNet():

    def test_ddpm_net_dimensions(self):
        model_dim = 64
        batch_size = 4
        image_size = 8
        time_embed = punet.GaussianFourierProjection(model_dim)
        time_block = punet.ResnetTimeBlock(model_dim, model_dim)
        resnet_block = punet.ResnetBlock(model_dim, model_dim)
        upsampler = punet.UpSampler(model_dim, model_dim//2)
        downsampler = punet.DownSampler(model_dim, 2*model_dim)
        t = -torch.log(torch.rand(batch_size))
        x = torch.randn(batch_size, model_dim, image_size, image_size)
        te = time_embed(t)
        assert (te.shape == torch.Size([batch_size, model_dim]))

        assert (
            time_block(te).shape == torch.Size([batch_size, model_dim, 1, 1])
        )

        assert (
            resnet_block(x, te).shape == torch.Size(
                [batch_size, model_dim, image_size, image_size]
            )
        )

        assert (
            upsampler(x).shape == torch.Size(
                [batch_size, model_dim//2, 2*image_size, 2*image_size]
            )
        )

        assert (
            downsampler(x).shape == torch.Size(
                [batch_size, 2*model_dim, image_size//2, image_size//2]
            )
        )

    def test_scheduler_dimensions(self):
        scheduler = diffsci.models.DDPMScheduler()
        t = scheduler.sample(10)
        assert (scheduler.calpha(t).shape == t.shape)
        assert (scheduler.sigma(t).shape == t.shape)
        assert (scheduler.beta(t).shape == t.shape)
        assert (scheduler.alpha(t).shape == t.shape)

    def test_training_conditional(self):
        MAINFOLDER = pathlib.Path("..")
        DATAPATH = MAINFOLDER/"saveddata"

        def destroy_image(x, scale_factor=2):
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

        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, mnist_dataset, scale_factor=4):
                self.mnist_dataset = mnist_dataset
                self.scale_factor = scale_factor

            def __len__(self):
                return len(self.mnist_dataset)

            def __getitem__(self, idx):
                x, _ = self.mnist_dataset[idx]
                y = destroy_image(x, self.scale_factor)
                return x, y

        batch_size = 4

        mnist_dataset = torchvision.datasets.MNIST(
            DATAPATH/'external',
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )

        dataset = CustomDataset(mnist_dataset)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True, num_workers=4
        )

        model = diffsci.models.PUNetCond(4)
        scheduler = diffsci.models.DDPMScheduler()
        trainer = diffsci.models.CondDDPMTrainer(model, scheduler, dataloader)
        trainer.set_optimizer_and_scheduler()


class TestMLPNet():
    def test_mlp_net_dimensions(self):
        x = torch.randn(101, 2)
        t = torch.linspace(0, 1, 101)
        y = torch.randn(101, 3)
        model_uncond = diffsci.models.MLPUncond(2)
        model_cond = diffsci.models.MLPCond(2, 3)
        assert (model_uncond(x, t).shape == torch.Size([101, 2]))
        assert (model_cond(x, t, y).shape == torch.Size([101, 2]))


if __name__ == "__main__":
    tester_PU = TestPUNet()
    tester_PU.test_ddpm_net_dimensions()
    tester_mlp = TestMLPNet()
    tester_mlp.test_mlp_net_dimensions()
    print("All tests passed")