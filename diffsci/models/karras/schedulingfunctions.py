import torch
from torch import Tensor
from jaxtyping import Float


class SchedulingFunctions(torch.nn.Module):
    constant_scaling_fn = False
    identity_noise_fn = False
    has_pf_score_multiplier = False
    has_pf_scale_multiplier = False

    def scaling_fn(self,
                   t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        raise NotImplementedError

    def scaling_fn_deriv(self,
                         t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        raise NotImplementedError

    def noise_fn(self,
                 t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        raise NotImplementedError

    def inverse_noise_fn(self,
                         t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        raise NotImplementedError

    def noise_fn_deriv(self,
                       t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        raise NotImplementedError

    def pf_score_multiplier(self,
                            t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        raise NotImplementedError

    def pf_scale_multiplier(self,
                            t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        raise NotImplementedError


class EDMSchedulingFunctions(SchedulingFunctions):
    constant_scaling_fn = True
    identity_noise_fn = True

    def scaling_fn(self,
                   t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        return 1 + 0*t

    def scaling_fn_deriv(self,
                         t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        return 0*t

    def noise_fn(self,
                 t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        return 1*t

    def inverse_noise_fn(self,
                         t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        return 1*t

    def noise_fn_deriv(self,
                       t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        return 1 + 0*t


class VPSchedulingFunctions(SchedulingFunctions):
    constant_scaling_fn = False  # TODO: Remove
    has_pf_score_multiplier = False  # TODO: Set to true
    has_pf_scale_multiplier = False  # TODO: Set to true

    def __init__(self,
                 beta_data: float = 19.9,
                 beta_min: float = 0.1):
        super().__init__()
        self.beta_data = beta_data
        self.beta_min = beta_min

    def scaling_fn(self,
                   t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        # return torch.exp(-t)
        # return 1.0 + 0.0*t
        expoent = 0.5*self.beta_data*t**2 + self.beta_min*t
        return torch.exp(-expoent/2)

    def scaling_fn_deriv(self,
                         t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        # return -torch.exp(-t)
        # return 0.0 + 0.0*t
        expoent = 0.5*self.beta_data*t**2 + self.beta_min*t
        expoent_deriv = self.beta_data*t + self.beta_min
        return -expoent_deriv/2*torch.exp(-expoent/2)

    def noise_fn(self,
                 t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        expoent = 0.5*self.beta_data*t**2 + self.beta_min*t
        return torch.sqrt(torch.exp(expoent)-1)

    def inverse_noise_fn(self,
                         t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        y = torch.log(t**2 + 1)
        delta = self.beta_min**2 + 2*self.beta_data*y
        return (-self.beta_min + torch.sqrt(delta))/self.beta_data

    def noise_fn_deriv(self,
                       t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        expoent = 0.5*self.beta_data*t**2 + self.beta_min*t
        expoent_deriv = self.beta_data*t + self.beta_min
        exponentiated = torch.exp(expoent)
        numerator = expoent_deriv*exponentiated
        denominator = 2*torch.sqrt(exponentiated-1)
        return numerator/denominator

    def pf_score_multiplier(self,
                            t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        # s^2 \sigma' \sigma
        return 1/2*(self.beta_data*t + self.beta_min)

    def pf_scale_multiplier(self,
                            t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        # s'/s = (\log s)'
        return -1/2*(self.beta_data*t + self.beta_min)


class VESchedulingFunctions(SchedulingFunctions):
    constant_scaling_fn = True
    has_pf_score_multiplier = True

    def scaling_fn(self,
                   t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        return 1 + 0*t

    def scaling_fn_deriv(self,
                         t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        return 0*t

    def noise_fn(self,
                 t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        return torch.sqrt(t)

    def inverse_noise_fn(self,
                         t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        return t**2

    def noise_fn_deriv(self,
                       t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        return 0.5/torch.sqrt(t)

    def pf_score_multiplier(self,
                            t: Float[Tensor, '...']) -> Float[Tensor, '...']:
        # \sigma' \sigma
        return 0.5 + 0*t


def name_to_scheduling_functions(
        name: str,
        *args,
        **kwargs
        ) -> SchedulingFunctions:  # noqa: F821
    possible_names = ["EDM", "VP", "VE"]
    if name not in possible_names:
        raise ValueError(f"Unknown scheduling function name: {name}")
    if name == "EDM":
        return EDMSchedulingFunctions()
    elif name == "VP":
        return VPSchedulingFunctions(*args, **kwargs)
    elif name == "VE":
        return VESchedulingFunctions(*args, **kwargs)
    else:
        raise ValueError(f"Unknown scheduling function name: {name}")
