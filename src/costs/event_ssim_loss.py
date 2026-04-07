import logging

import torch
from fused_ssim import fused_ssim

from . import CostBase

logger = logging.getLogger(__name__)


class EventSSIMLoss(CostBase):
    """Event image SSIM loss used in training."""

    name = "event_ssim_loss"
    required_keys = [
        "diff_img",
        "accumulated_events",
        "use_diff_image",
        "bayered_diff",
        "is_color",
    ]

    @CostBase.register_history  # type: ignore
    @CostBase.catch_key_error  # type: ignore
    def calculate(self, arg: dict) -> torch.Tensor:
        diff_img = arg["diff_img"]
        accumulated_events = arg["accumulated_events"]
        use_diff_image = arg["use_diff_image"]
        bayered_diff = arg["bayered_diff"]
        is_color = arg["is_color"]
        weight = arg.get("weight", 1.0)
        iwe_mask = arg.get("iwe_mask", None)

        if accumulated_events is None:
            raise ValueError("accumulated_events should not be None.")

        if use_diff_image and bayered_diff:
            if iwe_mask is None:
                raise ValueError("iwe_mask is required when bayered_diff is True.")
            diff_img_ssim = diff_img.to(dtype=torch.float32) * iwe_mask
        elif use_diff_image and is_color:
            diff_img_ssim = diff_img.to(dtype=torch.float32) / 4
        else:
            diff_img_ssim = diff_img.to(dtype=torch.float32)

        if is_color and use_diff_image:
            diff_img_ssim = diff_img_ssim.permute(0, 3, 1, 2)
        else:
            diff_img_ssim = diff_img_ssim.unsqueeze(0)

        accumulated_events_ssim = accumulated_events.to(dtype=torch.float32).detach()

        if is_color and use_diff_image:
            accumulated_events_ssim = accumulated_events_ssim.permute(0, 3, 1, 2)
        else:
            accumulated_events_ssim = accumulated_events_ssim.unsqueeze(0)

        max1 = float(diff_img_ssim.max().item())
        max2 = float(accumulated_events_ssim.max().item())
        normalization_val_p = max(max1, max2, 1e-8)
        min1 = float(diff_img_ssim.min().item())
        min2 = float(accumulated_events_ssim.min().item())
        normalization_val_n = max(-min1, -min2, 1e-8)

        diff_img_ssim_p = (diff_img_ssim / normalization_val_p).clamp(0.0, 1.0)
        accumulated_events_ssim_p = (
            accumulated_events_ssim / normalization_val_p
        ).clamp(0.0, 1.0)
        diff_img_ssim_n = (-diff_img_ssim / normalization_val_n).clamp(0.0, 1.0)
        accumulated_events_ssim_n = (
            -accumulated_events_ssim / normalization_val_n
        ).clamp(0.0, 1.0)

        if use_diff_image:
            ssim_loss = fused_ssim(
                diff_img_ssim_p, accumulated_events_ssim_p, padding="valid"
            ) + fused_ssim(diff_img_ssim_n, accumulated_events_ssim_n, padding="valid")
            ssim_loss = (2 - ssim_loss) * weight
        else:
            ssim_loss = fused_ssim(
                diff_img_ssim_p, accumulated_events_ssim_p, padding="valid"
            )
            ssim_loss = (1 - ssim_loss) * weight

        if self.direction in ["minimize", "natural"]:
            return ssim_loss
        logger.warning("The loss is specified as maximize direction")
        return -ssim_loss
