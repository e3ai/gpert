import numpy as np
import torch


def linlog(input: torch.Tensor, linlog_threshold: float = 20) -> torch.Tensor:
    """Piecewise linear-log transform used for event simulation."""
    lin_slope = np.log(linlog_threshold) / linlog_threshold
    return torch.where(input < linlog_threshold, input * lin_slope, torch.log(input))


__all__ = ["linlog"]
