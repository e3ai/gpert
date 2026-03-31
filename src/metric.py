"""Metric utilities aligned with robust_e_nerf evaluation pipeline."""

from typing import Optional, Union
import easydict
import numpy as np
import torch
import torchmetrics
import lpips


class Metric(torch.nn.Module):
    """Compute L1, PSNR, SSIM and LPIPS identical to robust_e_nerf."""

    METRIC_NAMES = ["l1", "psnr", "ssim", "lpips"]

    def __init__(self):
        super().__init__()
        self.lpips = lpips.LPIPS(net="alex").eval()
        self.lpips.requires_grad_(False)

    def init_batch_metric(self):
        return easydict.EasyDict({name: [] for name in self.METRIC_NAMES})

    def compute(
        self,
        pred_img,  # ([[batch_size,] 1/3,] img_height, img_width)
        target_img,  # ([[batch_size,] 1/3,] img_height, img_width)
        min_target_val,
        max_target_val,
    ):
        # Verify inputs
        assert pred_img.shape == target_img.shape
        assert 2 <= target_img.dim() <= 4
        if target_img.dim() > 2:
            assert target_img.shape[-3] in (1, 3)
        assert 0 <= min_target_val < max_target_val
        assert torch.all(min_target_val <= target_img) and torch.all(
            target_img <= max_target_val
        )

        # Normalize image dimensions
        if target_img.dim() < 4:
            new_shape = (4 - target_img.dim()) * (
                1,
            ) + target_img.shape  # (1, 1/3, img_height, img_width)
            pred_img = pred_img.view(*new_shape)  # (1, 1/3, img_height, img_width)
            target_img = target_img.view(*new_shape)  # (1, 1/3, img_height, img_width)

        # Compute metrics
        metric = easydict.EasyDict({})

        """
        NOTE:
            For monochrome images, `metric.l1` gives the mean L1 distance/loss
            across pixels (& batches). For RGB images, `metric.l1` gives
            1/3 * mean L1 distance/loss across pixels (& batches), which yields
            a comparable value compared to monochrome images.
        """
        metric.l1 = torch.nn.functional.l1_loss(input=pred_img, target=target_img)

        """
        NOTE:
            We set `data_range=target_val_range` as PSNR is a relative metric.
        """
        target_val_range = max_target_val - min_target_val
        metric.psnr = torchmetrics.functional.psnr(
            preds=pred_img,
            target=target_img,
            data_range=target_val_range,
            reduction="elementwise_mean",
            dim=(1, 2, 3),
        )

        """
        NOTE:
            We set `data_range=max_target_val` as SSIM is an absolute metric.
        """
        metric.ssim = torchmetrics.functional.ssim(
            preds=pred_img,
            target=target_img,
            data_range=max_target_val,
            reduction="elementwise_mean",
        )

        # Normalize both predicted & target image values equally so that target
        # image values are in [-1, 1] & convert monochrome images to RGB, if
        # necessary, for LPIPS computation
        pred_img = (
            2 * (pred_img - min_target_val) / target_val_range - 1
        )  # (batch_size, 1/3, img_height, img_width)
        target_img = (
            2 * (target_img - min_target_val) / target_val_range - 1
        )  # (batch_size, 1/3, img_height, img_width)
        pred_img = pred_img.expand(
            -1, 3, -1, -1
        )  # (batch_size,   3, img_height, img_width)
        target_img = target_img.expand(
            -1, 3, -1, -1
        )  # (batch_size,   3, img_height, img_width)
        metric.lpips = self.lpips(in0=pred_img, in1=target_img).mean()

        return metric


def eval_metrics(
    metric: Metric,
    pred_img_color: torch.Tensor,
    gt_img_color: torch.Tensor,
    mask_np: Optional[np.ndarray],
    use_mean_bg: bool = False,
    device: Union[str, torch.device] = "cpu",
    min_val: float = 0.001953125,  # 0.5/256
    max_val: float = 0.998046875,  # 1 - 0.5/256
) -> tuple:
    """Evaluate PSNR/SSIM/LPIPS between prediction and ground truth."""
    if pred_img_color.shape != gt_img_color.shape:
        raise ValueError("pred_img_color and gt_img_color must share the same shape")

    pred_img_color = pred_img_color.to(torch.float32)
    gt_img_color = gt_img_color.to(torch.float32)
    pred_img_color = torch.clamp(pred_img_color, min=min_val, max=max_val)

    gt_eval = gt_img_color

    if pred_img_color.ndim == 2:
        pred_img_color = pred_img_color.unsqueeze(-1)
        gt_eval = gt_eval.unsqueeze(-1)

    pred_cf = pred_img_color.permute(2, 0, 1).unsqueeze(0)
    gt_cf = gt_eval.permute(2, 0, 1).unsqueeze(0)

    metric_out = metric.compute(
        pred_img=pred_cf,
        target_img=gt_cf,
        min_target_val=min_val,
        max_target_val=max_val,
    )

    return (
        float(metric_out.psnr.detach().item()),
        float(metric_out.ssim.detach().item()),
        float(metric_out.lpips.detach().item()),
    )
