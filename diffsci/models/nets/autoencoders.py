import torch
import diffusers

import diffsci.models


class AutoencoderKLWrapper(torch.nn.Module):
    def __init__(self, vae, channels=1):
        super().__init__()
        self.vae = vae
        self.inference = False
        self.channels = channels

    def expand_channels(self, x):
        shape = list(x.shape)
        shape[-3] = 3
        print(x.shape[-3], self.channels)
        if self.channels == 1:
            return x.expand(*shape)
        elif self.channels == 2:
            y = torch.zeros(shape)
            y[..., :2, :, :] = x
            return y
        elif self.channels == 3:
            return x
        else:
            raise ValueError(f"Invalid number of channels: {self.channels}")

    def squeeze_channels(self, x):
        if self.channels == 1:
            return x.mean(dim=-3, keepdim=True)
        elif self.channels == 2:
            return x[..., :2, :, :]
        elif self.channels == 3:
            return x
        else:
            raise ValueError(f"Invalid number of channels: {self.channels}")

    def forward(self, x, has_batch_dim=True):
        res = self.decode(self.encode(x, has_batch_dim), has_batch_dim)
        return res

    def encode(self, x, has_batch_dim=True):
        if not has_batch_dim:
            x = x.unsqueeze(0)
        res = self.vae.encode(self.expand_channels(x))['latent_dist'].sample()
        if not has_batch_dim:
            res = res[0]
        return res

    def decode(self, z, has_batch_dim=True):
        if not has_batch_dim:
            z = z.unsqueeze(0)
        res = self.squeeze_channels(self.vae.decode(z)['sample'])
        if not has_batch_dim:
            res = res[0]
        return res


class OurAutoencoderKLWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.inference = False

    def expand_channels(self, x):
        shape = list(x.shape)
        shape[-3] = 3
        return x.expand(*shape)

    def squeeze_channels(self, x):
        return x.mean(dim=-3, keepdim=True)

    def forward(self, x, has_batch_dim=True):
        res = self.decode(self.encode(x, has_batch_dim), has_batch_dim)
        return res

    def encode(self, x, has_batch_dim=True, mode=False):
        if not has_batch_dim:
            x = x.unsqueeze(0)
        if mode:
            res = self.vae.encode(x).latent_dist.mode()
        else:
            res = self.vae.encode(x).latent_dist.sample()
        if not has_batch_dim:
            res = res[0]
        return res

    def decode(self, z, has_batch_dim=True):
        if not has_batch_dim:
            z = z.unsqueeze(0)
        res = self.vae.decode(z).sample
        if not has_batch_dim:
            res = res[0]
        return res


class LDMAutoencoderKLWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.inference = False

    def expand_channels(self, x):
        shape = list(x.shape)
        shape[-3] = 3
        return x.expand(*shape)

    def squeeze_channels(self, x):
        return x.mean(dim=-3, keepdim=True)

    def forward(self, x, has_batch_dim=True):
        res = self.decode(self.encode(x, has_batch_dim).sample(), has_batch_dim)
        return res

    def encode(self, x, has_batch_dim=True, mode=False):
        if not has_batch_dim:
            x = x.unsqueeze(0)
        if mode:
            res = self.vae.encode(x).mode()
        else:
            res = self.vae.encode(x).sample()
        if not has_batch_dim:
            res = res[0]
        return res

    def decode(self, z, has_batch_dim=True):
        if not has_batch_dim:
            z = z.unsqueeze(0)
        res = self.vae.decode(z)
        if not has_batch_dim:
            res = res[0]
        return res


class AutoencoderTinyWrapper(torch.nn.Module):
    def __init__(self, vae, channels=1):
        super().__init__()
        self.vae = vae
        self.inference = False
        self.channels = channels

    def expand_channels(self, x):
        shape = list(x.shape)
        shape[-3] = 3
        print(x.shape[-3], self.channels)
        if self.channels == 1:
            return x.expand(*shape)
        elif self.channels == 2:
            y = torch.zeros(shape)
            y[..., :2, :, :] = x
            return y
        elif self.channels == 3:
            return x
        else:
            raise ValueError(f"Invalid number of channels: {self.channels}")

    def squeeze_channels(self, x):
        if self.channels == 1:
            return x.mean(dim=-3, keepdim=True)
        elif self.channels == 2:
            return x[..., :2, :, :]
        elif self.channels == 3:
            return x
        else:
            raise ValueError(f"Invalid number of channels: {self.channels}")

    def forward(self, x, has_batch_dim=True):
        res = self.decode(self.encode(x, has_batch_dim), has_batch_dim)
        return res

    def encode(self, x, has_batch_dim=True):
        if not has_batch_dim:
            x = x.unsqueeze(0)
        res = self.vae.encode(self.expand_channels(x))['latents']
        if not has_batch_dim:
            res = res[0]
        return res

    def decode(self, z, has_batch_dim=True):
        if not has_batch_dim:
            z = z.unsqueeze(0)
        res = self.squeeze_channels(self.vae.decode(z)['sample'])
        if not has_batch_dim:
            res = res[0]
        return res


def load_autoencoder(type: str,
                     path: str = None):
    if type == 'kl1':
        url = ("https://huggingface.co/stabilityai/"
               "sd-vae-ft-mse-original/blob/main/"
               "vae-ft-mse-840000-ema-pruned.safetensors")
        vae_base_model = (diffusers.
                          AutoencoderKL.
                          from_single_file(url))
        vae_wrapper = AutoencoderKLWrapper(vae_base_model)
    elif type == 'tiny1':
        vae_base_model = (diffusers.
                          AutoencoderTiny.
                          from_pretrained("madebyollin/taesdxl",
                                          torch_dtype=torch.float32))
        vae_wrapper = AutoencoderTinyWrapper(vae_base_model)
    elif type == 'our_kl':
        vae_base_model = (diffsci.models.AutoencoderKL(
            in_channels=1,
            out_channels=1,
            ).from_pretrained(path, torch_dtype=torch.float32))
        vae_wrapper = OurAutoencoderKLWrapper(vae_base_model)
    else:
        raise ValueError(f"Unknown autoencoder type: {type}")
    return vae_wrapper
