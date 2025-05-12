import math

import torch
from torch import Tensor
from torch.distributions import Normal
from jaxtyping import Shaped, Float

from diffsci.torchutils import broadcast_from_below
from diffsci.global_constants import SUM_STABILIZER


class AnalyticalDataset(torch.utils.data.Dataset):
    r"""

    An abstract class for toy datasets that we can calculate analytically
    following the expression:

    p(x;\sigma) = \int N(x;x_0, \sigma) p(x_0) dx_0.
    """

    def __init__(self, num_samples: int, *args, **kwargs):
        self.num_samples = num_samples
        self.samples = self.sample()

    def sample(self
               ) -> Shaped[Tensor, "num_samples *shape"]:  # noqa: F821
        """
        Make a num_samples quantity of samples out of the desired distribution.
        """
        raise NotImplementedError

    def logprob(self,
                x: Shaped[Tensor, "num_samples *shape"],  # noqa: F821
                sigma: Shaped[Tensor, "num_samples"]  # noqa: F821
                ) -> Float[Tensor, "num_samples"]:  # noqa: F821
        """
        Calculate the log-probability of x.

        Parameters:
        ----------
        x : torch.Tensor of shape (nbatch, ...).
        sigma : torch.Tensor of shape (nbatch).

        Returns:
        -------
        logp : torch.Tensor of shape (nbatch,).
        """
        raise NotImplementedError

    def gradlogprob(self,
                    x: Shaped[Tensor, "num_samples *shape"],  # noqa: F821
                    sigma: Shaped[Tensor, "num_samples"]  # noqa: F821
                    ) -> Float[Tensor, "num_samples *shape"]:  # noqa: F821
        """
        Calculate the grad log-probability of x.

        Parameters:
        ----------
        x : torch.Tensor of shape (nbatch, *shape).
        sigma : torch.Tensor of shape (nbatch).

        Returns:
        -------
        grad_logp : torch.Tensor of shape (nbatch, *shape).
        """
        raise NotImplementedError

    def optimal_denoiser_predictor(
            self,
            x: Float[Tensor, "batch *shape"],  # noqa: F821
            sigma: Float[Tensor, "batch"],  # noqa: F821
            scale: None | Float[Tensor, "batch"] = None  # noqa: F821
            ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        raise NotImplementedError

    def denoiser(self,
                 x: Shaped[Tensor, "num_samples *shape"],  # noqa: F821
                 sigma: Shaped[Tensor, "num_samples"]  # noqa: F821
                 ) -> Float[Tensor, "num_samples *shape"]:  # noqa: F821
        """
        Calculates the analytical denoising function.
        """

        gradlogprob = self.gradlogprob(x, sigma)  # [b, *shape]
        sigma_broadcasted = broadcast_from_below(sigma, x)   # [b, *shape]
        return x + sigma_broadcasted**2 * gradlogprob

    def optimal_noise_predictor(
            self,
            x: Float[Tensor, "batch *shape"],  # noqa: F821
            sigma: Float[Tensor, "batch"],  # noqa: F821
            scale: None | Float[Tensor, "batch"] = None  # noqa: F821
            ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        x0 = self.optimal_denoiser_predictor(x, sigma, scale=scale)
        if scale is not None:
            scale_ = broadcast_from_below(scale, x0)
            x0 = x0*scale_
        sigma_ = broadcast_from_below(sigma, x0)
        return (x - x0)/sigma_

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generate a sample from the dataset.
        """
        return self.samples[idx]


class SinglePointDataset(AnalyticalDataset):
    """
    A dataset consisting of a generator of a single point. We can define it
    mathematically as the Dirac delta distribution in x_0.

    Parameters:
        ----------
        x0 : float | torch.Tensor of shape [*shape].
            The point to generate.
        shape : list[int].
            The dimension of the space.
        num_samples : int.
            The number of points to generate before considering the dataset
            'complete'.
    """
    def __init__(self,
                 num_samples: int,
                 x0: Float[Tensor, "*shape"],  # noqa: F821
                 ):
        self.shape = x0.shape
        self.x0 = x0
        super().__init__(num_samples)

    def sample(self
               ) -> Shaped[Tensor, "num_samples *shape"]:  # noqa: F821
        """
        Make a num_samples quantity of samples out of the desired distribution.
        In this case, it consists of num_sample copies of x0.
        """
        return self.x0.expand(self.num_samples, *self.shape)

    def logprob(self,
                x: Shaped[Tensor, "batch *shape"],  # noqa: F821
                sigma: Shaped[Tensor, "batch"]  # noqa: F821
                ) -> Float[Tensor, "batch"]:  # noqa: F821
        """
        Calculate the log-probability of x.

        Parameters:
        ----------
        x : torch.Tensor of shape (nbatch, ...).
        sigma : torch.Tensor of shape (nbatch).

        Returns:
        -------
        logp : torch.Tensor of shape (nbatch,).
        """
        sigma = broadcast_from_below(sigma, x)
        diff = (x - self.x0)**2
        sqnorm = (torch.sum(diff, dim=tuple(range(1, diff.dim()))))**2
        expterm = -0.5 * sqnorm / sigma**2
        ndim = sum(self.x0.shape)
        normalizer = -ndim/2*torch.log(2*math.pi*sigma**2)
        logp = expterm + normalizer

        return logp

    def gradlogprob(self,
                    x: Float[Tensor, "batch *shape"],  # noqa: F821
                    sigma: Float[Tensor, "batch"]  # noqa: F821
                    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        """
        Calculate the grad log-probability of x.

        Parameters:
        ----------
        x : torch.Tensor of shape (nbatch, *shape).
        sigma : torch.Tensor of shape (nbatch).

        Returns:
        -------
        grad_logp : torch.Tensor of shape (nbatch, *shape).
        """
        sigma = broadcast_from_below(sigma, x)
        grad_logp = -(x - self.x0) / (sigma ** 2)

        return grad_logp

    def optimal_denoiser_predictor(
            self,
            x: Float[Tensor, "batch *shape"],  # noqa: F821
            sigma: Float[Tensor, "batch"],  # noqa: F821
            scale: None | Float[Tensor, "batch"] = None  # noqa: F821
            ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        return self.x0.unsqueeze(0).expand(*x.shape)


class SingleGaussianDataset(AnalyticalDataset):
    """
    A dataset consisting of a generator of a single multi-variate gaussian
    with diagonal covariance matrix equal to scale * I.

    Parameters:
        ----------
        x0 : float | torch.Tensor of shape [*shape].
            Mean of the gaussian distribution.
        scale: float.
            Standard deviation of each component of the multi-variate gaussian
            distribution.
        shape : list[int].
            The dimension of the space.
        num_samples : int.
            The number of points to generate.
    """
    def __init__(self,
                 num_samples: int,
                 x0: Float[Tensor, "*shape"],  # noqa: F821
                 scale: float = 1.0,
                 ):
        self.shape = x0.shape
        self.x0 = x0
        self.scale = scale
        super().__init__(num_samples)

    def sample(self
               ) -> Shaped[Tensor, "num_samples *shape"]:  # noqa: F821
        mean = self.x0.expand(self.num_samples, *self.shape)
        noise = self.scale * torch.randn_like(mean)
        return mean + noise

    def logprob(self,
                x: Shaped[Tensor, "batch *shape"],  # noqa: F821
                sigma: Shaped[Tensor, "batch"]  # noqa: F821
                ) -> Float[Tensor, "batch"]:  # noqa: F821
        """
        Calculate the log-probability of x.

        Parameters:
        ----------
        x : torch.Tensor of shape (nbatch, ...).
        sigma : torch.Tensor of shape (nbatch).

        Returns:
        -------
        logp : torch.Tensor of shape (nbatch,).
        """
        sigma_mod = torch.sqrt(sigma**2 + self.scale**2)
        diff = (x - self.x0)**2
        sqnorm = (torch.sum(diff, dim=tuple(range(1, diff.dim()))))
        expterm = -0.5 * sqnorm / sigma_mod**2
        ndim = sum(self.x0.shape)
        normalizer = -ndim/2*torch.log(2*math.pi*sigma_mod**2)
        logp = expterm + normalizer

        return logp

    def gradlogprob(self,
                    x: Float[Tensor, "batch *shape"],  # noqa: F821
                    sigma: Float[Tensor, "batch"]  # noqa: F821
                    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        """
        Calculate the grad log-probability of x.

        Parameters:
        ----------
        x : torch.Tensor of shape (nbatch, *shape).
        sigma : torch.Tensor of shape (nbatch).

        Returns:
        -------
        grad_logp : torch.Tensor of shape (nbatch, *shape).
        """
        sigma = broadcast_from_below(sigma, x)
        sigma_mod = torch.sqrt(sigma**2 + self.scale**2)
        grad_logp = -(x - self.x0) / (sigma_mod ** 2)

        return grad_logp


class ZeroDataset(SinglePointDataset):
    """
    A dataset consisting of a single zero point.
    """
    def __init__(self, num_samples: int, shape: list[int]):
        super().__init__(num_samples, torch.zeros(shape))


class ZeroMeanGaussianDataset(SingleGaussianDataset):
    """
    A dataset of a single multi-variate gaussian distribution with zero mean.
    """
    def __init__(self,
                 num_samples: int,
                 shape: list[int],
                 scale: float = 1.0):
        super().__init__(num_samples,
                         torch.zeros(shape),
                         scale=scale)


class MixtureOfPointsDataset(AnalyticalDataset):
    """
    A dataset which is a mixture of single points datasets. In more common
    terms, it can be viewed as a discrete probability distribution.

    Parameters:
        ----------
        shape: list[int].
            The dimension of the space.
        points: float | torch.Tensor of shape [nmixtures *shape].
            The set of points of which we should sample.
        weights: float | torch.Tensor of shape [nmixtures *shape].
            The set of weights associated to each point when sampling.
        num_samples: int.
            The number of points to generate.
    """

    def __init__(self,
                 num_samples: int,
                 points: Float[Tensor, "nmixtures *shape"],  # noqa: F821
                 weights: Float[Tensor, "nmixtures"]):  # noqa: F821
        self.points = points
        self.weights = weights/torch.sum(weights)  # We guarantee normalization
        super().__init__(num_samples)

    def sample(self
               ) -> Shaped[Tensor, "num_samples *shape"]:  # noqa: F821
        indexes = torch.multinomial(self.weights,
                                    self.num_samples,
                                    replacement=True)
        return self.points[indexes, ...]

    def gradlogprob(self,
                    x: Float[Tensor, "batch *shape"],  # noqa: F821
                    sigma: Float[Tensor, "batch"]  # noqa: F821
                    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        """
        Calculate the grad log-probability of x.

        Parameters:
        ----------
        x : torch.Tensor of shape (nbatch, *shape).
        sigma : torch.Tensor of shape (nbatch).

        Returns:
        -------
        grad_logp : torch.Tensor of shape (nbatch, *shape).
        """
        x = x.unsqueeze(1)  # [b, 1, *shape]
        p = self.points.unsqueeze(0)  # [1, n, *shape]
        diff = (x - p)  # [b, n, *shape]
        sumdims = tuple(range(2, diff.dim()))
        norm2 = torch.sum((diff**2), dim=sumdims)  # [b, n]
        expfactors = torch.exp(-0.5*norm2/(sigma[:, None]**2))  # [b, n]
        wfactors = expfactors * self.weights  # [b, n]
        sigma_ = broadcast_from_below(sigma, diff)  # [b, n, *shape]
        terms = -diff/(sigma_**2)  # [b, n, *shape]
        wfactors = broadcast_from_below(wfactors, terms)  # [b, n, *shape]
        wfactors = wfactors + SUM_STABILIZER  # [b, n, *shape]
        wfactors = wfactors/wfactors.sum(dim=1, keepdim=True)  # [b, n, *shape]
        grad_logp = (wfactors*terms).sum(dim=1)  # [b, *shape]
        return grad_logp

    def optimal_denoiser_predictor(
            self,
            x: Float[Tensor, "batch *shape"],  # noqa: F821
            sigma: Float[Tensor, "batch"],  # noqa: F821
            scale: None | Float[Tensor, "batch"] = None  # noqa: F821
            ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        x = x.unsqueeze(1)  # [b, 1, *shape]
        p = self.points.unsqueeze(0)  # [n, *shape]
        if scale is not None:
            scale_ = broadcast_from_below(scale, p)  # [batch, 1, *shape]
            p = p*scale_  # [batch, n, shape]
        diff = (x - p)  # [b, n, *shape]
        sumdims = tuple(range(2, diff.dim()))
        norm2 = torch.sum((diff**2), dim=sumdims)  # [b, n]
        sigma_ = broadcast_from_below(sigma, norm2)
        scores = -1/(2*sigma_**2) * norm2  # [b, n]
        scores = scores + torch.log(self.weights)  # [b, n]
        #  [b, n]
        scores = scores - torch.logsumexp(scores, dim=1, keepdim=True)
        factors = torch.exp(scores)  # [b, n]
        factors = broadcast_from_below(factors, diff)  # [b, n, *shape]
        return (factors*p).sum(dim=1)  # [b, *shape]


class MixtureOfGaussiansDataset(AnalyticalDataset):
    """
    A dataset which is a mixture of gaussian datasets.

    Parameters:
        ----------
        shape: list[int].
            The dimension of the space.
        means: float | torch.Tensor of shape [nmixtures *shape].
            The set of points of which we should sample.
        scale: float | torch.Tensor of shape [nmixtures *shape]
            The set of standard deviations for each distribution
        weights: float | torch.Tensor of shape [nmixtures *shape].
            The set of weights associated to each distribution when sampling.
        num_samples: int.
            The number of points to generate.
    """
    def __init__(self,
                 num_samples: int,
                 means: Float[Tensor, "nmixtures *shape"],  # noqa: F821
                 weights: Float[Tensor, "nmixtures"],  # noqa: F821
                 scale: float | Float[
                     Tensor, "nmixtures"] = 1.0):  # noqa: F821
        self.means = means
        self.weights = weights/torch.sum(weights)  # We guarantee normalization
        self.scale = scale
        super().__init__(num_samples)

    def sample(self
               ) -> Shaped[Tensor, "num_samples *shape"]:  # noqa: F821
        indexes = torch.multinomial(self.weights,
                                    self.num_samples,
                                    replacement=True)
        means = self.means[indexes, ...]
        if isinstance(self.scale, float):
            scale = self.scale
        else:
            scale = self.scale[indexes].unsqueeze(-1)
        noise = scale * torch.randn_like(means)
        return means + noise

    def prob(self,
             x: Float[Tensor, "batch *shape"],  # noqa: F821
             sigma: Float[Tensor, "batch"],  # noqa: F821
             scaling: Float[Tensor, "batch"] = 1.0 # noqa: F821
             ) -> Float[Tensor, "batch"]:  # noqa: F821
        """
        Calculate the probability of x.
        Parameters:
        ----------
        x : torch.Tensor of shape (nbatch, *shape).
        sigma : torch.Tensor of shape (nbatch).
        Returns:
        -------
        prob : torch.Tensor of shape (nbatch,).
        """
        if isinstance(self.scale, float):
            scale = self.scale
        else:
            scale = self.scale.unsqueeze(0)
        sigma_mod = scaling * torch.sqrt(sigma.unsqueeze(-1)**2 + scale**2)  # [b, n]
        x = x.unsqueeze(1)  # [b, 1, *shape]
        p = scaling * self.means.unsqueeze(0)  # [1, n, *shape]
        diff = (x - p)  # [b, n, *shape]
        sumdims = tuple(range(2, diff.dim()))
        norm2 = torch.sum((diff**2), dim=sumdims)  # [b, n]
        expfactors = torch.exp(-0.5*norm2/(sigma_mod**2))  # [b, n]
        wfactors = expfactors * self.weights  # [b, n]
        dims = x.shape[1:]
        n = 1
        for dim in dims:
            n *= dim
        # print(f'Dimension: {n}')
        normalizer = 1/(2*math.pi*sigma_mod**2)**(n/2)  # [b, n]
        wfactors = wfactors * normalizer  # [b, n]
        prob = wfactors.sum(dim=1)  # [b]
        return prob

    def gradlogprob(self,
                    x: Float[Tensor, "batch *shape"],  # noqa: F821
                    sigma: Float[Tensor, "batch"]  # noqa: F821
                    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        """
        Calculate the grad log-probability of x.

        Parameters:
        ----------
        x : torch.Tensor of shape (nbatch, *shape).
        sigma : torch.Tensor of shape (nbatch).

        Returns:
        -------
        grad_logp : torch.Tensor of shape (nbatch, *shape).
        """
        if isinstance(self.scale, float):
            scale = self.scale
        else:
            scale = self.scale.unsqueeze(0)     # [1, n]
        dims = x.shape[1:]
        n = 1
        for dim in dims:
            n *= dim
        sigma_mod = torch.sqrt(sigma.unsqueeze(-1)**2 + scale**2)  # [b, n]
        x = x.unsqueeze(1)  # [b, 1, *shape]
        p = self.means.unsqueeze(0)  # [1, n, *shape]
        diff = (x - p)  # [b, n, *shape]
        sumdims = tuple(range(2, diff.dim()))
        norm2 = torch.sum((diff**2), dim=sumdims)  # [b, n]
        expfactors = torch.exp(-0.5*norm2/(sigma_mod**2))  # [b, n]
        wfactors = expfactors * self.weights  # [b, n]
        sigma_mod_ = broadcast_from_below(sigma_mod, diff)  # [b, n, *shape]
        terms = -diff/(sigma_mod_**2)  # [b, n, *shape]
        const = 2*math.pi*sigma_mod**2  # [b, n]
        const = const**(-n/2)  # [b, n]
        wfactors = wfactors * const  # [b, n]
        wfactors = broadcast_from_below(wfactors, terms)  # [b, n, *shape]
        wfactors = wfactors + SUM_STABILIZER  # [b, n, *shape]
        wfactors = wfactors/wfactors.sum(dim=1, keepdim=True)  # [b, n, *shape]
        grad_logp = (wfactors*terms).sum(dim=1)  # [b, *shape]
        return grad_logp


class DiagonalGaussianDataset(AnalyticalDataset):
    """
    A dataset consisting of a generator of a multivariate gaussian with
    diagonal covariance matrix.

    Parameters
        ----------
        x0 : float | torch.Tensor of shape [*shape]
            The mean of the gaussian
        shape : list[int]
            The dimension of the space
        num_samples : int
            The number of points to generate before considering the dataset
            'complete'
        diag_std : float | torch.Tensor of shape [*shape]
            The sqrt of the diagonal of the covariance matrix.
    """
    def __init__(self,
                 num_samples: int,
                 x0: Float[Tensor, "*shape"],  # noqa: F821
                 diag_std: Float[Tensor, "*shape"]):    # noqa: F821

        self.shape = x0.shape
        self.x0 = x0
        self.std = diag_std
        super().__init__(num_samples)

    def sample(self
               ) -> Shaped[Tensor, "num_samples *shape"]:  # noqa: F821
        mean = self.x0.expand(self.num_samples, *self.shape)
        noise = self.std * torch.randn_like(mean)
        return mean + noise

    def gradlogprob(self,
                    x: Float[Tensor, "batch *shape"],  # noqa: F821
                    sigma: Float[Tensor, "batch"]  # noqa: F821
                    ) -> Float[Tensor, "batch *shape"]:  # noqa: F821
        """
        Calculate the grad log-probability of x perturbed by noise of variance
        sigma**2.

        Parameters
        ----------
        x : torch.Tensor of shape (nbatch, *shape)
        sigma : torch.Tensor of shape (nbatch)

        Returns
        -------
        grad_logp : torch.Tensor of shape (nbatch, *shape)
        """
        sigma = broadcast_from_below(sigma, x)
        sigma_mod = torch.sqrt(sigma**2 + self.std**2)
        grad_logp = -(x - self.x0) / (sigma_mod ** 2)

        return grad_logp


class Single1DUniformDataset(AnalyticalDataset):
    """
    A dataset consisting of a generator of a single univariate uniform distribution in the interval [a,b].

    Parameters:
        ----------
        num_samples : int.
            The number of points to generate.
        interval : list[float].
            A list of two floats representing the lower (a) and upper (b) bounds of the uniform distribution.
    """
    def __init__(self,
                 num_samples: int,
                 interval: list[float]):
        self.a, self.b = interval
        super().__init__(num_samples)

    def sample(self) -> Tensor:  # noqa: F821
        samples = torch.rand(self.num_samples, 1)
        return samples * (self.b - self.a) + self.a

    def prob(self,
             x: Tensor,  # noqa: F821
             sigma: Tensor  # noqa: F821
             ) -> Tensor:  # noqa: F821
        """
        Calculate the probability of x.

        Parameters:
        ----------
        x : torch.Tensor of shape (nbatch, 1).
        sigma : torch.Tensor of shape (nbatch).

        Returns:
        -------
        p : torch.Tensor of shape (nbatch,).
        """
        a = self.a
        b = self.b
        normal_dist = Normal(0, 1)
        phi_a = normal_dist.cdf((x - a) / sigma)
        phi_b = normal_dist.cdf((x - b) / sigma)
        p = 1 / (b - a) * (phi_a - phi_b)
        return p

    def gradlogprob(self,
                    x: Tensor,  # noqa: F821
                    sigma: Tensor,  # noqa: F821
                    epsilon: float = 1e-15
                    ) -> Tensor:  # noqa: F821
        """
        Calculate the grad log-probability of x.

        Parameters:
        ----------
        x : torch.Tensor of shape (nbatch, 1).
        sigma : torch.Tensor of shape (nbatch).

        Returns:
        -------
        grad_logp : torch.Tensor of shape (nbatch, 1).
        """
        a = self.a
        b = self.b
        normal_dist = Normal(0, 1)
        sigma = broadcast_from_below(sigma, x)

        pdf_a = normal_dist.log_prob((x - a) / sigma).exp()
        pdf_b = normal_dist.log_prob((x - b) / sigma).exp()
        phi_a = normal_dist.cdf((x - a) / sigma)
        phi_b = normal_dist.cdf((x - b) / sigma)

        num = pdf_a - pdf_b
        den = phi_a - phi_b + epsilon  # Add epsilon to avoid division by zero
        grad_logp = num / (den * sigma)
        return grad_logp


class MixtureOf1DUniformsDataset(AnalyticalDataset):
    """
    A dataset consisting of a generator of a mixture of univariate uniform distributions.

    Parameters:
        ----------
        num_samples : int.
            The number of points to generate.
        intervals : list[list[float]].
            A list of intervals, where each interval is a list [a, b].
        weights : list[float].
            Mixture weights for each uniform distribution (should sum to 1).
    """
    def __init__(self,
                 num_samples: int,
                 intervals: list[list[float]],
                 weights: list[float]):
        self.intervals = intervals
        self.weights = torch.tensor(weights) / sum(weights)  # Normalize the weights
        assert len(intervals) == len(weights), "Number of intervals must match number of weights"
        super().__init__(num_samples)

    def sample(self) -> Tensor:  # noqa: F821
        mixture_indices = torch.multinomial(self.weights, self.num_samples, replacement=True)
        samples = torch.zeros(self.num_samples, 1)
        for i in range(len(self.intervals)):
            mask = (mixture_indices == i)
            num_samples_i = mask.sum()
            if num_samples_i > 0:
                a, b = self.intervals[i]
                samples[mask] = torch.rand(num_samples_i, 1) * (b - a) + a
        return samples

    def prob(self,
             x: Tensor,  # noqa: F821
             sigma: Tensor  # noqa: F821
             ) -> Tensor:  # noqa: F821
        """
        Calculate the probability of x for the mixture distribution.

        Parameters:
        ----------
        x : torch.Tensor of shape (nbatch, 1).
        sigma : torch.Tensor of shape (nbatch).

        Returns:
        -------
        p : torch.Tensor of shape (nbatch,).
        """
        normal_dist = Normal(0, 1)
        sigma = broadcast_from_below(sigma, x)
        total_prob = torch.zeros(x.shape)

        # Sum the probabilities from each uniform component in the mixture
        for i, (a, b) in enumerate(self.intervals):
            phi_a = normal_dist.cdf((x - a) / sigma)
            phi_b = normal_dist.cdf((x - b) / sigma)
            prob_i = 1 / (b - a) * (phi_a - phi_b)
            total_prob += self.weights[i] * prob_i

        total_prob = total_prob.squeeze(-1)
        return total_prob

    def gradlogprob(self,
                    x: Tensor,  # noqa: F821
                    sigma: Tensor,  # noqa: F821
                    epsilon: float = 1e-15
                    ) -> Tensor:  # noqa: F821
        """
        Calculate the grad log-probability of x for the mixture distribution.

        Parameters:
        ----------
        x : torch.Tensor of shape (nbatch, 1).
        sigma : torch.Tensor of shape (nbatch).

        Returns:
        -------
        grad_logp : torch.Tensor of shape (nbatch, 1).
        """
        normal_dist = Normal(0, 1)
        sigma = broadcast_from_below(sigma, x)
        total_p = torch.zeros(x.shape)
        total_gradp = torch.zeros(x.shape)

        # Sum the gradients from each uniform component in the mixture
        for i, (a, b) in enumerate(self.intervals):
            pdf_a = normal_dist.log_prob((x - a) / sigma).exp()
            pdf_b = normal_dist.log_prob((x - b) / sigma).exp()
            phi_a = normal_dist.cdf((x - a) / sigma)
            phi_b = normal_dist.cdf((x - b) / sigma)

            gradp = (pdf_a - pdf_b) / (b-a)
            p = (phi_a - phi_b) / (b-a)
            total_gradp += self.weights[i] * gradp
            total_p += self.weights[i] * p

        grad_logp = total_gradp / (total_p * sigma + epsilon)
        return grad_logp
