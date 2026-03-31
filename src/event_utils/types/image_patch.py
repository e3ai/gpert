import copy
from dataclasses import dataclass
from typing import Any, List

import numpy as np


@dataclass
class ImagePatch:
    """Dataclass for patch"""

    # center of coordinates
    x: np.int16
    y: np.int16
    size: int  # Currently width = height

    @property
    def x_min(self) -> int:
        return self.x - np.floor(self.size / 2)

    @property
    def x_max(self) -> int:
        return self.x + np.floor(self.size / 2)

    @property
    def y_min(self) -> int:
        return self.y - np.floor(self.size / 2)

    @property
    def y_max(self) -> int:
        return self.y + np.floor(self.size / 2)

    def surface_lines(self) -> List:
        return [self.surface_line(i) for i in range(0, 4)]

    def surface_line(self, index: int) -> np.ndarray:
        if index == 0:
            return np.array([[self.x_max, self.y_min], [self.x_max, self.y_max]])
        elif index == 1:
            return np.array([[self.x_max, self.y_max], [self.x_min, self.y_max]])
        elif index == 2:
            return np.array([[self.x_min, self.y_max], [self.x_min, self.y_min]])
        elif index == 3:
            return np.array([[self.x_min, self.y_min], [self.x_max, self.y_min]])
        raise ValueError

    def surface_unit(self, index: int) -> np.ndarray:
        if index == 0:
            return np.array([1, 0])
        elif index == 1:
            return np.array([0, 1])
        elif index == 2:
            return np.array([-1, 0])
        elif index == 3:
            return np.array([0, -1])
        raise ValueError

    def copy(self) -> Any:
        return copy.deepcopy(self)
