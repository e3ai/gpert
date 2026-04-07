import copy
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Event:
    """Dataclass for events"""

    # x-y pixel value in python image coordinates
    # (0, 0),     .....        (width, 0)
    # ...         (x, y),      ...
    # (0, height) .....        (width, height)
    x: np.int16
    y: np.int16
    timestamp: np.float64
    polarity: bool  # true for positive, false for negative

    @property
    def p(self) -> int:
        return self.polarity

    @property
    def t(self) -> np.float64:
        return self.timestamp

    @property
    def color(self) -> tuple:
        if self.p:
            return (255, 0, 0)  # Red
        else:
            return (0, 0, 255)  # Blue

    def copy(self) -> Any:
        return copy.deepcopy(self)
