from typing import Tuple

import numpy as np
import torch


def gamma_correction(
    pred_img,
    gt_img,
    mask=None,
    log_eps=0.00196078431,
    device="cpu",
    scale_offset=None,
    clip_percentiles=(1.0, 99.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match predicted intensities to ground truth via log-domain affine fit."""

    if not isinstance(pred_img, torch.Tensor) or not isinstance(gt_img, torch.Tensor):
        raise TypeError("pred_img and gt_img must be torch.Tensor instances")

    def _to_cbhw(img: torch.Tensor) -> tuple[torch.Tensor, str, torch.dtype]:
        original_dtype = img.dtype
        layout: str

        if img.ndim == 2:
            layout = "hw"
            img_cbhw = img.unsqueeze(0).unsqueeze(0)
        elif img.ndim == 3:
            if img.shape[0] in (1, 3):
                layout = "chw"
                img_cbhw = img.unsqueeze(1)
            elif img.shape[-1] in (1, 3):
                layout = "hwc"
                img_cbhw = img.permute(2, 0, 1).unsqueeze(1)
            else:
                raise ValueError("3D tensors must have channel dimension of size 1 or 3")
        elif img.ndim == 4:
            if img.shape[0] in (1, 3) and img.shape[1] not in (1, 3):
                layout = "cbhw"
                img_cbhw = img
            elif img.shape[1] in (1, 3):
                layout = "bchw"
                img_cbhw = img.permute(1, 0, 2, 3)
            elif img.shape[-1] in (1, 3):
                layout = "bhwc"
                img_cbhw = img.permute(3, 0, 1, 2)
            else:
                raise ValueError("4D tensors must have channel dimension of size 1 or 3")
        else:
            raise ValueError("Unsupported tensor dimensionality for gamma correction")

        return img_cbhw.to(torch.float32), layout, original_dtype

    def _from_cbhw(
        img: torch.Tensor, layout: str, dtype: torch.dtype, ref_input: torch.Tensor
    ) -> torch.Tensor:
        if layout == "hw":
            restored = img.squeeze(0).squeeze(0)
        elif layout == "hwc":
            restored = img.squeeze(1).permute(1, 2, 0)
        elif layout == "chw":
            restored = img.squeeze(1)
        elif layout == "bchw":
            restored = img.permute(1, 0, 2, 3)
        elif layout == "bhwc":
            restored = img.permute(1, 2, 3, 0)
        elif layout == "cbhw":
            restored = img
        else:
            raise ValueError(f"Unknown layout tag: {layout}")

        if (
            ref_input.ndim == 3
            and ref_input.shape[-1] == 1
            and layout in {"hwc", "bhwc"}
        ):
            restored = restored[..., 0]
            restored = restored.unsqueeze(-1)

        restored = restored.to(dtype if dtype.is_floating_point else torch.float32)
        return restored

    pred_cbhw, pred_layout, pred_dtype = _to_cbhw(pred_img)
    gt_cbhw, _, _ = _to_cbhw(gt_img)

    C, B, H, W = pred_cbhw.shape
    if C not in (1, 3):
        raise ValueError("Images must have 1 or 3 channels")

    if scale_offset is not None:
        pred_log = (pred_cbhw + log_eps).log()
        pred_flat = torch.nn.functional.pad(
            pred_log.reshape(C, B * H * W, 1), (0, 1), value=1.0
        )
        pred_flat = pred_flat.to(device=device, dtype=torch.float64)
        aligned = (
            (pred_flat @ scale_offset)
            .to(device="cuda", dtype=torch.float32)
            .squeeze(-1)
        )
        aligned = aligned.view(C, B, H, W)
        corrected = aligned.exp()

        corrected = _from_cbhw(corrected, pred_layout, pred_dtype, pred_img)

        return corrected, scale_offset

    pred_log = (pred_cbhw + log_eps).log()
    gt_log = (gt_cbhw + log_eps).log()

    pred_flat = torch.nn.functional.pad(
        pred_log.reshape(C, B * H * W, 1), (0, 1), value=1.0
    )
    gt_flat = gt_log.reshape(C, B * H * W, 1)

    pred_flat = pred_flat.to(device=device, dtype=torch.float64)
    gt_flat = gt_flat.to(device=device, dtype=torch.float64)

    scale_offset, *_ = torch.linalg.lstsq(pred_flat, gt_flat)

    aligned = (
        (pred_flat @ scale_offset).to(device="cuda", dtype=torch.float32).squeeze(-1)
    )
    del pred_flat, gt_flat

    aligned = aligned.view(C, B, H, W)
    corrected = aligned.exp()
    del aligned

    corrected = _from_cbhw(corrected, pred_layout, pred_dtype, pred_img)

    return corrected, scale_offset


def solve_normal_equations(preds_logs, imgs_gt_log, mask=None):
    """Solve y ≈ a*x + b via normal equations per channel."""
    pred_np = preds_logs.detach().cpu().numpy()
    gt_np = imgs_gt_log.detach().cpu().numpy()

    if pred_np.ndim == 3 and gt_np.ndim == 3:
        if pred_np.shape[-1] not in (1, 3):
            pred_np = pred_np[..., None]
            gt_np = gt_np[..., None]
    elif pred_np.ndim == 2 and gt_np.ndim == 2:
        pred_np = pred_np[..., None]
        gt_np = gt_np[..., None]

    assert pred_np.shape[-1] == gt_np.shape[-1], "Channel count mismatch between preds and gt"
    C = pred_np.shape[-1]

    spatial_shape = pred_np.shape[:-1]
    x_all = pred_np.reshape(-1, C)
    y_all = gt_np.reshape(-1, C)

    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = np.asarray(mask)

        if mask_np.shape == spatial_shape:
            m = mask_np
        elif mask_np.shape == spatial_shape + (1,):
            m = mask_np[..., 0]
        else:
            try:
                m = np.broadcast_to(mask_np, spatial_shape)
            except Exception:
                m = mask_np.reshape(-1)
        m = m != 0
        m_flat = m.reshape(-1)
        if m_flat.ndim != 1 or m_flat.shape[0] != x_all.shape[0]:
            m_flat = None
    else:
        m_flat = None

    a_list = []
    b_list = []
    for c in range(C):
        x = x_all[:, c].astype(np.float64)
        y = y_all[:, c].astype(np.float64)

        if m_flat is not None:
            valid = m_flat.astype(bool)
            if valid.any():
                x = x[valid]
                y = y[valid]

        X = np.column_stack([np.ones_like(x), x])

        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        b_c, a_c = float(beta[0]), float(beta[1])

        if not np.isfinite(a_c):
            a_c = 5.0
        if not np.isfinite(b_c):
            b_c = 5.0

        a_list.append(a_c)
        b_list.append(b_c)

    if C == 1:
        return a_list[0], b_list[0]

    return np.array(a_list, dtype=np.float64), np.array(b_list, dtype=np.float64)


__all__ = ["gamma_correction", "solve_normal_equations"]
