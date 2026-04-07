from dataclasses import dataclass
from typing import List

import numpy as np

from . import Event


@dataclass
class EventBatch:
    """List of events."""

    events: List[Event]

    def __getitem__(self, index):
        if type(index) is int:
            return self.events[index]
        else:
            return EventBatch(events=list(np.array(self.events)[index]))

    def __len__(self) -> int:
        return self.n

    def clear(self) -> None:
        self.events = []

    def add(self, e: Event):
        self.events.append(e)

    @property
    def n(self) -> int:
        return len(self.events)

    @property
    def x(self) -> np.ndarray:
        return np.array([event.x for event in self.events], dtype=np.int16)

    @property
    def y(self) -> np.ndarray:
        return np.array([event.y for event in self.events], dtype=np.int16)

    @property
    def t(self) -> np.ndarray:
        return np.array([event.t for event in self.events], dtype=np.float64)

    @property
    def p(self) -> np.ndarray:
        return np.array([event.p for event in self.events], dtype=np.int8)

    @property
    def color(self) -> np.ndarray:
        # (3 x n), RGB
        return np.stack([event.color for event in self.events]).astype(np.uint8)

    @property
    def color_plt(self) -> np.ndarray:
        # (3 x n), RGB
        colormap = np.array(["b", "r"])
        return colormap[self.p]

    def asarray(self) -> np.ndarray:
        """Returns (n_events, 4) ... (x, y, t, p)"""
        return np.vstack([self.x, self.y, self.t, self.p]).astype(np.float64).T
