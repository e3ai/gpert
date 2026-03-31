from typing import Tuple, Union

import numpy as np
import torch

from .event import Event
from .event_batch import EventBatch
from .flow_patch import FlowPatch
from .image_patch import ImagePatch
from .polarity import Polarity

NUMPY_TORCH = Union[np.ndarray, torch.Tensor]
FLOAT_TORCH = Union[float, torch.Tensor]

# Trans, Rotation
REL_POSE = Tuple[np.ndarray, np.ndarray]


def to_numpy(arr: NUMPY_TORCH):
    if isinstance(arr, torch.Tensor):
        return arr.clone().detach().cpu().numpy()
    return arr
