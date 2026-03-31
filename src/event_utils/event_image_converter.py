import logging
from typing import Tuple, Union

import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms.functional import gaussian_blur

from .types import FLOAT_TORCH, NUMPY_TORCH

logger = logging.getLogger(__name__)


class EventImageConverter(object):
    """Convert events into frame representations used by the current pipeline."""

    def __init__(
        self, image_size: tuple, outer_padding: Union[int, Tuple[int, int]] = 0
    ):
        if isinstance(outer_padding, (int, float)):
            self.outer_padding = (int(outer_padding), int(outer_padding))
        else:
            self.outer_padding = outer_padding
        self.image_size = tuple(
            int(i + p * 2) for i, p in zip(image_size, self.outer_padding)
        )

    def create_eventframe(
        self,
        events: NUMPY_TORCH,
        method: str = "bilinear_vote",
        weight: Union[float, NUMPY_TORCH, FLOAT_TORCH] = 1.0,
        sigma: int = 1,
    ) -> NUMPY_TORCH:
        """Create an event frame from numpy or torch events."""
        if isinstance(events, torch.Tensor):
            return self.create_image_from_events_tensor(
                events, method, weight=weight, sigma=sigma
            )
        elif isinstance(events, np.ndarray):
            return self.create_image_from_events_numpy(
                events, method, weight=weight, sigma=sigma
            )

        e = f"Non-supported type of events. {type(events)}"
        logger.error(e)
        raise RuntimeError(e)

    def create_image_from_events_numpy(
        self,
        events: np.ndarray,
        method: str = "bilinear_vote",
        weight: Union[float, np.ndarray] = 1.0,
        sigma: int = 1,
    ) -> np.ndarray:
        """Create an event image from numpy events."""
        if method == "bilinear_vote":
            image = self.bilinear_vote_numpy(events, weight=weight)
        elif method == "polarity":
            pos_flag = events[..., 3] > 0
            if isinstance(weight, np.ndarray):
                pos_image = self.bilinear_vote_numpy(
                    events[pos_flag], weight=weight[pos_flag]
                )
                neg_image = self.bilinear_vote_numpy(
                    events[~pos_flag], weight=weight[~pos_flag]
                )
            else:
                pos_image = self.bilinear_vote_numpy(events[pos_flag], weight=weight)
                neg_image = self.bilinear_vote_numpy(events[~pos_flag], weight=weight)
            image = np.stack([pos_image, neg_image], axis=-3)
        else:
            e = f"{method = } is not supported."
            logger.error(e)
            raise NotImplementedError(e)

        if sigma > 0:
            image = gaussian_filter(image, sigma)
        return image

    def create_image_from_events_tensor(
        self,
        events: torch.Tensor,
        method: str = "bilinear_vote",
        weight: FLOAT_TORCH = 1.0,
        sigma: int = 0,
    ) -> torch.Tensor:
        """Create an event image from torch events."""
        if method == "bilinear_vote":
            padded_events = pad_sequence(events, batch_first=True, padding_value=-10000)
            image = self.bilinear_vote_tensor(padded_events, weight=weight)
        elif method == "polarity":
            pos_flag = events[..., 3] > 0
            filtered_pos_events = [e[m] for e, m in zip(events, pos_flag)]
            padded_pos_events = pad_sequence(
                filtered_pos_events, batch_first=True, padding_value=-10000
            )
            filtered_neg_events = [e[m] for e, m in zip(events, ~pos_flag)]
            padded_neg_events = pad_sequence(
                filtered_neg_events, batch_first=True, padding_value=-10000
            )

            if isinstance(weight, torch.Tensor):
                pos_image = self.bilinear_vote_tensor(
                    padded_pos_events, weight=weight[pos_flag]
                )
                neg_image = self.bilinear_vote_tensor(
                    padded_neg_events, weight=weight[~pos_flag]
                )
            else:
                pos_image = self.bilinear_vote_tensor(padded_pos_events, weight=weight)
                neg_image = self.bilinear_vote_tensor(padded_neg_events, weight=weight)

            image = torch.stack([pos_image, neg_image], axis=-3)
        else:
            e = f"{method = } is not implemented"
            logger.error(e)
            raise NotImplementedError(e)

        if sigma > 0:
            if len(image.shape) == 2:
                image = image[None, None, ...]
            elif len(image.shape) == 3:
                image = image[:, None, ...]
            image = gaussian_blur(image, kernel_size=3, sigma=sigma)
        return torch.squeeze(image)

    def bilinear_vote_numpy(
        self, events: np.ndarray, weight: Union[float, np.ndarray] = 1.0
    ) -> np.ndarray:
        """Accumulate events into image by bilinear voting (numpy)."""
        if isinstance(weight, np.ndarray):
            assert weight.shape == events.shape[:-1]
        if len(events.shape) == 2:
            events = events[None, ...]

        ph, pw = self.outer_padding
        h, w = self.image_size
        nb = len(events)
        image = np.zeros((nb, h * w), dtype=np.float64)

        floor_xy = np.floor(events[..., :2] + 1e-8)
        floor_to_xy = events[..., :2] - floor_xy

        x1 = floor_xy[..., 1] + pw
        y1 = floor_xy[..., 0] + ph
        inds = np.concatenate(
            [
                x1 + y1 * w,
                x1 + (y1 + 1) * w,
                (x1 + 1) + y1 * w,
                (x1 + 1) + (y1 + 1) * w,
            ],
            axis=-1,
        )
        inds_mask = np.concatenate(
            [
                (0 <= x1) * (x1 < w) * (0 <= y1) * (y1 < h),
                (0 <= x1) * (x1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
                (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1) * (y1 < h),
                (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
            ],
            axis=-1,
        )
        w_pos0 = (1 - floor_to_xy[..., 0]) * (1 - floor_to_xy[..., 1]) * weight
        w_pos1 = floor_to_xy[..., 0] * (1 - floor_to_xy[..., 1]) * weight
        w_pos2 = (1 - floor_to_xy[..., 0]) * floor_to_xy[..., 1] * weight
        w_pos3 = floor_to_xy[..., 0] * floor_to_xy[..., 1] * weight
        vals = np.concatenate([w_pos0, w_pos1, w_pos2, w_pos3], axis=-1)
        inds = (inds * inds_mask).astype(np.int64)
        vals = vals * inds_mask
        for i in range(nb):
            np.add.at(image[i], inds[i], vals[i])
        return image.reshape((nb,) + self.image_size).squeeze()

    def bilinear_vote_tensor(
        self, events: torch.Tensor, weight: FLOAT_TORCH = 1.0
    ) -> torch.Tensor:
        """Accumulate events into image by bilinear voting (torch)."""
        if len(events.shape) == 2:
            events = events.unsqueeze(0)

        ph, pw = self.outer_padding
        h, w = self.image_size
        nb = events.shape[0]
        image = events.new_zeros((nb, h * w))

        floor_xy = torch.floor(events[..., :2] + 1e-6)
        floor_to_xy = events[..., :2] - floor_xy
        floor_xy = floor_xy.long()

        x1 = floor_xy[..., 1] + pw
        y1 = floor_xy[..., 0] + ph
        inds = torch.cat(
            [
                x1 + y1 * w,
                x1 + (y1 + 1) * w,
                (x1 + 1) + y1 * w,
                (x1 + 1) + (y1 + 1) * w,
            ],
            dim=-1,
        )
        inds_mask = torch.cat(
            [
                (0 <= x1) * (x1 < w) * (0 <= y1) * (y1 < h),
                (0 <= x1) * (x1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
                (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1) * (y1 < h),
                (0 <= x1 + 1) * (x1 + 1 < w) * (0 <= y1 + 1) * (y1 + 1 < h),
            ],
            axis=-1,
        )

        w_pos0 = (1 - floor_to_xy[..., 0]) * (1 - floor_to_xy[..., 1]) * weight
        w_pos1 = floor_to_xy[..., 0] * (1 - floor_to_xy[..., 1]) * weight
        w_pos2 = (1 - floor_to_xy[..., 0]) * floor_to_xy[..., 1] * weight
        w_pos3 = floor_to_xy[..., 0] * floor_to_xy[..., 1] * weight
        vals = torch.cat([w_pos0, w_pos1, w_pos2, w_pos3], dim=-1)

        vals = vals.to(events)
        inds = (inds * inds_mask).long()
        image = image.to(dtype=events.dtype)
        vals = vals * inds_mask
        image.scatter_add_(1, inds, vals)
        return image.reshape((nb,) + self.image_size).squeeze()
