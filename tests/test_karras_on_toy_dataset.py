import torch
import lightning

import diffsci.models
import diffsci.data


def test_karras_on_zero_dataset(dim=1, nsteps=100, nsamples=100):
    dataset = diffsci.data.ZeroDataset(num_samples=256*50, shape=[dim])
    noise_scheduler = diffsci.models.EDMScheduler()

    x = torch.randn(nsamples, dim)

    def gradlogprob(x, t):
        logprob = dataset.gradlogprob(x, t)
        return logprob

    history = noise_scheduler.propagate_backward(x,
                                                 gradlogprob,
                                                 nsteps,
                                                 record_history=True)
    assert (history.shape == torch.Size([nsteps+1, nsamples, dim]))
    assert (torch.isclose(history[0], x)).all()
    assert (torch.isclose(history[-1],
                          torch.tensor(0.0),
                          rtol=1e-2,
                          atol=1e-2).all())

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, x, t):
            return dataset.denoiser(x, t) + 0.0*self.dummy*x

    x = torch.randn(nsamples, dim)
    model = ToyModel()
    config = diffsci.models.KarrasModuleConfig.from_edm()
    module = diffsci.models.KarrasModule(model, config)
    config.preconditioner = diffsci.models.NullPreconditioner()
    samples = module.propagate_white_noise(x,
                                           nsteps=nsteps,
                                           record_history=False)
    assert (samples.shape == torch.Size([nsamples, dim]))
    assert ((torch.abs(samples) < 1e-2).all())
    samples = module.sample(nsamples, shape=[dim])
    assert ((torch.abs(samples) < 1e-2).all())
    history = module.propagate_white_noise(x,
                                           record_history=True,
                                           nsteps=nsteps)
    assert (history.shape == torch.Size([nsteps+1, nsamples, dim]))
    assert torch.isclose(history[0],
                         x*module.config.noisescheduler.maximum_scale).all()
    assert (torch.isclose(history[-1],
                          torch.tensor(0.0),
                          rtol=1e-2,
                          atol=1e-2).all())

    batch_size = 8
    train_size = 256*45
    test_size = 256*5
    train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                [train_size,
                                                                 test_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=4)

    trainer = lightning.Trainer(max_epochs=5,
                                enable_checkpointing=False,
                                logger=False,
                                accelerator="cpu",
                                fast_dev_run=True)
    trainer.fit(model=module,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    assert ((torch.isclose(trainer.logged_metrics['train_loss'],
                           torch.tensor(0.0))).all())

    model = diffsci.models.MLPUncond(dim, [20])
    module.model = model
    trainer.fit(model=module,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)


if __name__ == "__main__":
    test_karras_on_zero_dataset()
    test_karras_on_zero_dataset(dim=3, nsteps=1000, nsamples=10)

    print("All tests passed!")
