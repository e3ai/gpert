import logging

import cv2
import numpy as np
import torch

from . import GradientMagnitude

logger = logging.getLogger(__name__)


class GradientMagnitudeHuber(GradientMagnitude):
    """Gradient Magnitude Huber loss for less impact on the larger accumulation in IWE.

    Args:
        direction (str) ... 'minimize' or 'maximize' or `natural`.
            Defines the objective function. If natural, it returns more interpretable value.
    """

    name = "gradient_magnitude_huber"
    required_keys = ["iwe", "omit_boundary"]

    def __init__(
        self,
        direction="minimize",
        store_history: bool = False,
        cuda_available=False,
        precision="32",
        *args,
        **kwargs,
    ):
        super().__init__(
            direction=direction,
            store_history=store_history,
            cuda_available=cuda_available,
            precision=precision,
        )

    def calculate_torch(self, iwe: torch.Tensor, omit_boundary: bool) -> torch.Tensor:
        if len(iwe.shape) == 2:
            iwe = iwe[None, None, ...]
        elif len(iwe.shape) == 3:
            iwe = iwe[:, None, ...]
        if self.precision == "64":
            iwe = iwe.double()
        iwe_sobel = self.torch_sobel.forward(iwe) / 8.0
        # iwe_sobel = self.torch_sobel.forward(iwe) / 2.0
        gx = iwe_sobel[:, 0]
        gy = iwe_sobel[:, 1]
        if omit_boundary:
            gx = gx[..., 1:-1, 1:-1]
            gy = gy[..., 1:-1, 1:-1]
        magnitude = torch.mean(torch.abs(gx) + torch.abs(gy))
        if self.direction == "minimize":
            return -magnitude
        return magnitude

    def calculate_numpy(self, iwe: np.ndarray, omit_boundary: bool) -> float:
        """Calculate contrast of the count image.
        Inputs:
            iwe (np.ndarray) ... [W, H]. Image of warped events
            omit_bondary (bool) ... Omit boundary if True.

        Returns:
            (float) ... magnitude of gradient.
        """
        gx = cv2.Sobel(iwe, cv2.CV_64F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(iwe, cv2.CV_64F, 0, 1, ksize=3) / 8.0
        if omit_boundary:
            gx = gx[..., 1:-1, 1:-1]
            gy = gy[..., 1:-1, 1:-1]
        magnitude = np.mean(np.abs(gx) + np.abs(gy))
        if self.direction == "minimize":
            return -magnitude
        return magnitude
