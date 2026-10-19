"""Shared count-distribution constructors for active scLAMP-SR models."""

import torch
from torch.distributions import NegativeBinomial


NEGATIVE_BINOMIAL_PARAMETERIZATION = "pytorch_mean_dispersion_v1"


def negative_binomial_from_mean_dispersion(mean, dispersion, eps=1e-6):
    """Create a PyTorch negative binomial with mean ``mean``.

    PyTorch parameterizes the distribution as
    ``E[X] = total_count * probs / (1 - probs)``.  Setting
    ``probs = mean / (mean + dispersion)`` therefore gives
    ``E[X] = mean`` and ``Var[X] = mean + mean**2 / dispersion``.
    """
    mean = torch.clamp_min(mean, eps)
    dispersion = torch.clamp_min(dispersion, eps)
    probs = mean / (mean + dispersion)
    probs = torch.clamp(probs, min=eps, max=1.0 - eps)
    return NegativeBinomial(total_count=dispersion, probs=probs)
