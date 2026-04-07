from dataclasses import dataclass
import torch


@dataclass
class Events:
    """Represents a collection of events.

    Attributes:
        t: A tensor of timestamps. ((b,) N)
        x: A tensor of x-coordinates. ((b,) N)
        y: A tensor of y-coordinates. ((b,) N)
        p: A tensor of polarities. ((b,) N)
    """

    t: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    p: torch.Tensor
