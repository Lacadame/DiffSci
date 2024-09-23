# Denoising Diffusion Probabilistic Models

The first work to propose a diffusion framework is the 2015 article 
[Deep Unsupervised Learning using Nonequilibrium Thermodynamics](
https://arxiv.org/abs/1503.03585) by Sohl-Dickstein et al. Despite being
a good theoretical introduction, you will prefer the two articles below if
you wish to just understand what we implemented here:

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
by Ho et al.;
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) by 
Song et al.

These two papers basically used the idea proposed by Sohl-Dickstein and gave it
a more practical and result-oriented approach. This is why the DDPM and DDIM here
presented are, in the most part, using the parameters proposed by them. By getting
the theoretical backgroud off the way, we can now guide you through the code.

## v1 x v2

This one is simple, ddpm_v1 is just a deprecated version of the ddpm before the team 
started to use pytorch lightning. This being the case, this readme will not delve
into the specifics of ddpm_v1. It is still here for historical reasons but
its code has not been revised for a long time and therefore should never be used.
Please prefer ddpm_v2 for any practical application.

## DDPMConfig Object

Diffusion models have so many parameters and hyperparameters that we opted to 
create a single class to configure at least the integrator, the scheduler and 
the loss metric we will use for every training. This also comes with a handful
of class methods for quick initialization of the most famous models cited in the
articles we provided above.

## Integrators

After reading the three articles above, it becomes clear that DDPM is about slowly
inserting noise into an object in a controlled way, so that after a large amount
of steps, the final distribution is indistinguishable from $\mathcal{N} (0, I)$.
The trick is that we train a U-net to learn how to slowly remove this noise, so that
we can generate samples from the data probability distribution by slowly removing
noise from new samples of $\mathcal{N}(0, I)$.

The Integrator is just the algorithm you use to slowly add and remove noise, the
name is inspired by a subsequent work of Karras et al., where he sees diffusion
models from the Stochastic Differential Equation perspective. We have also implemented
Karras' models in this repository and a better explanation can be found in its README.

## Schedulers

We guess there is no way to explain the scheduler without going a little bit deep 
into the math (which can already be found in the articles), so we will limit ourselves
to saying that the DDPM scheduler should not be confused with the learning rate
scheduler (which is a first honest confusion). The latter will be always referenced
as `lr_scheduler` in the code.