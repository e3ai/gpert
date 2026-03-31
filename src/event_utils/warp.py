import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch

from .types import FLOAT_TORCH, NUMPY_TORCH

logger = logging.getLogger(__name__)


def calculate_reftime(
    events: NUMPY_TORCH, direction: Union[str, float] = "first"
) -> FLOAT_TORCH:
    """Calculate reference time for warping."""
    if type(direction) is float:
        if isinstance(events, np.ndarray):
            per = events[..., 2].max(axis=-1) - events[..., 2].min(axis=-1)
            return events[..., 2].min(axis=-1) + per * direction
        elif isinstance(events, torch.Tensor):
            per = events[..., 2].max(-1).values - events[..., 2].min(-1).values
            return events[..., 2].min(-1).values + per * direction
    elif direction == "first":
        if isinstance(events, np.ndarray):
            return events[..., 2].min(axis=-1)
        elif isinstance(events, torch.Tensor):
            return torch.min(events[..., 2], -1).values
    elif direction == "middle":
        return calculate_reftime(events, 0.5)
    elif direction == "last":
        if isinstance(events, np.ndarray):
            return events[..., 2].max(axis=-1)
        elif isinstance(events, torch.Tensor):
            return events[..., 2].max(-1).values
    elif direction == "random":
        return calculate_reftime(events, np.random.uniform(low=0.0, high=1.0))
    elif direction == "before":
        return calculate_reftime(events, -1.0)
    elif direction == "after":
        return calculate_reftime(events, 2.0)

    e = (
        "direction argument should be first, middle, last. "
        f"Or float. {direction} is {type(direction)}"
    )
    logger.error(e)
    raise ValueError(e)


def calculate_dt(
    event: NUMPY_TORCH,
    reference_time: FLOAT_TORCH,
    time_period: Optional[FLOAT_TORCH] = None,
    normalize_t: bool = False,
) -> NUMPY_TORCH:
    """Calculate dt = t - reference_time with optional normalization."""
    dt = event[..., 2] - reference_time

    if not normalize_t:
        return dt

    if time_period is None:
        if isinstance(dt, np.ndarray):
            time_period = dt.max(axis=-1) - dt.min(axis=-1)
        elif isinstance(dt, torch.Tensor):
            time_period = dt.max(-1).values - dt.min(-1).values

    if isinstance(dt, np.ndarray):
        assert time_period is not None
        if isinstance(time_period, np.ndarray):
            return dt / time_period[..., None]
        return dt / float(time_period)

    if isinstance(dt, torch.Tensor):
        assert time_period is not None
        if isinstance(time_period, torch.Tensor):
            if time_period.ndim == 0:
                return dt / time_period
            return dt / time_period[..., None]
        return dt / float(time_period)

    return dt


def warp_event_by_event(
    events: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    z: torch.Tensor,
    f: torch.Tensor,
    image_size: Tuple[int, int],
    random_coeff: float,
    direction: Union[str, float] = "first",
    data_type: str = "opengl",
    device: torch.device = torch.device("cuda"),
    normalize_iwe: bool = False,
    normalize_t: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Warp events from camera motion and per-event depth.

    Args:
        events: [B, N, 4] or [N, 4], event order (x, y, t, p).
        v: [B, 3] translational velocity.
        w: [B, 3] angular velocity.
        z: [B, N] per-event depth.
        f: focal length scalar tensor.
    """
    del random_coeff, data_type

    x = events[..., 0]
    y = events[..., 1]
    t = events[..., 2]
    p = events[..., 3]

    x_flow = x - image_size[1] / 2.0
    y_flow = y - image_size[0] / 2.0
    weight = torch.ones_like(events[..., 0], device=device, dtype=torch.float32)

    flow_x_trans = 1.0 / z * (-f * v[:, 0:1] + x_flow * v[:, 2:3])
    flow_y_trans = 1.0 / z * (-f * v[:, 1:2] + y_flow * v[:, 2:3])
    flow_x_rot = (
        1.0
        / f
        * (
            x_flow * y_flow * w[:, 0:1]
            - (torch.square(f) + torch.square(x_flow)) * w[:, 1:2]
            + f * y_flow * w[:, 2:3]
        )
    )
    flow_y_rot = (
        1.0
        / f
        * (
            (torch.square(f) + torch.square(y_flow)) * w[:, 0:1]
            - x_flow * y_flow * w[:, 1:2]
            - f * x_flow * w[:, 2:3]
        )
    )
    flow_x = flow_x_trans + flow_x_rot
    flow_y = flow_y_trans + flow_y_rot

    mask = z == 1e-6
    flow_x = flow_x.masked_fill(mask, 0.0)
    flow_y = flow_y.masked_fill(mask, 0.0)

    ref_time = calculate_reftime(events, direction)
    dt = calculate_dt(events, ref_time, normalize_t=normalize_t)

    assert isinstance(dt, torch.Tensor)

    warped_x = x - dt * flow_x
    warped_y = y - dt * flow_y
    warped_event = torch.stack([warped_x, warped_y, t, p], dim=-1).to(
        dtype=torch.float32
    )

    if normalize_iwe:
        flow_magnitude = torch.sqrt(torch.square(flow_x) + torch.square(flow_y))
        weight = (weight / (flow_magnitude * dt + 1e-2)).to(events)

    if warped_event.ndim == 2:
        warped_event = warped_event.unsqueeze(0)

    return warped_event, weight
