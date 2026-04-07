from typing import Union
import numpy as np
import torch
from .trajectory import QuaternionWXYZ, Trajectory
from .intrinsics import Intrinsics
from .events import Events

NUMPY_TORCH = Union[np.ndarray, torch.Tensor]
FLOAT_TORCH = Union[float, torch.Tensor]
