import logging

import torch
import torch.nn.functional as F

from . import CostBase

logger = logging.getLogger(__name__)


class EventL1Loss(CostBase):
    """Event image L1 loss used in training."""

    name = "event_l1_loss"
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
        normalize = arg.get("normalize", False)
        use_masked = arg.get("use_masked", False)
        mask_weight = arg.get("mask_weight", 0.9)
        weight = arg.get("weight", 1.0)
        iwe_mask = arg.get("iwe_mask", None)

        if accumulated_events is None:
            raise ValueError("accumulated_events should not be None.")

        if normalize:
            max_val = torch.maximum(diff_img.max(), accumulated_events.max())
            min_val = torch.minimum(diff_img.min(), accumulated_events.min())
            scale = (max_val - min_val).clamp(min=1e-8)
            diff_img_l1 = diff_img / scale
            accumulated_events_l1 = accumulated_events / scale
        else:
            diff_img_l1 = diff_img
            accumulated_events_l1 = accumulated_events

        accumulated_events_l1 = accumulated_events_l1.detach()

        if use_diff_image:
            if bayered_diff:
                if iwe_mask is None:
                    raise ValueError("iwe_mask is required when bayered_diff is True.")
                diff_img_l1 = diff_img_l1 * iwe_mask
            elif is_color:
                diff_img_l1 = diff_img_l1 / 4

        l1loss_raw = F.l1_loss(diff_img_l1, accumulated_events_l1, reduction="none")

        if use_masked:
            mask = (accumulated_events != 0).float()
            weighted_loss = mask_weight * (l1loss_raw * mask).sum() / (
                mask.sum() + 1e-8
            ) + (1 - mask_weight) * (l1loss_raw * (1 - mask)).sum() / (
                (1 - mask).sum() + 1e-8
            )
            l1loss = weighted_loss
        else:
            l1loss = l1loss_raw.mean()

        l1loss = l1loss * weight

        if self.direction in ["minimize", "natural"]:
            return l1loss
        logger.warning("The loss is specified as maximize direction")
        return -l1loss
