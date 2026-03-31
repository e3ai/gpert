from typing import Optional, Tuple

import torch


def rgb_to_bayer_mosaic(
    image: torch.Tensor,
    bayer_pattern: str = "RGGB",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project RGB image(s) onto a Bayer mosaic defined by the configured pattern."""
    mask_rgb = torch.ones_like(image)
    if not isinstance(image, torch.Tensor):
        return mask_rgb, image

    if image.ndim not in (3, 4) or image.shape[-1] != 3:
        return mask_rgb, image

    pattern = bayer_pattern
    pattern_map: dict[tuple[int, int], Optional[int]] = {
        (0, 0): None,
        (0, 1): None,
        (1, 0): None,
        (1, 1): None,
    }

    match pattern:
        case "RGGB":
            pattern_map[(0, 0)] = 0
            pattern_map[(0, 1)] = 1
            pattern_map[(1, 0)] = 1
            pattern_map[(1, 1)] = 2
        case "BGGR":
            pattern_map[(0, 0)] = 2
            pattern_map[(0, 1)] = 1
            pattern_map[(1, 0)] = 1
            pattern_map[(1, 1)] = 0
        case "GRBG":
            pattern_map[(0, 0)] = 1
            pattern_map[(0, 1)] = 0
            pattern_map[(1, 0)] = 2
            pattern_map[(1, 1)] = 1
        case "GBRG":
            pattern_map[(0, 0)] = 1
            pattern_map[(0, 1)] = 2
            pattern_map[(1, 0)] = 0
            pattern_map[(1, 1)] = 1
        case _:
            raise ValueError(f"Unsupported Bayer pattern: {pattern}")

    if image.ndim == 3:
        image = image.unsqueeze(0)

    assert image.ndim == 4
    batch_size, height, width, _ = image.shape
    mosaic = torch.zeros(
        (batch_size, height, width), dtype=image.dtype, device=image.device
    )

    row_parity = (
        (torch.arange(height, device=image.device) % 2)
        .unsqueeze(1)
        .expand(height, width)
    )
    col_parity = (
        (torch.arange(width, device=image.device) % 2)
        .unsqueeze(0)
        .expand(height, width)
    )

    rgb_flat = image.view(batch_size, -1, 3)
    mosaic_flat = mosaic.view(batch_size, -1)

    for (r_parity, c_parity), channel_idx in pattern_map.items():
        if channel_idx is None:
            continue
        mask = ((row_parity == r_parity) & (col_parity == c_parity)).reshape(-1)
        if mask.any():
            mosaic_flat[:, mask] = rgb_flat[:, mask, channel_idx]
        mask_rgb[..., channel_idx] = mask.reshape(batch_size, height, width)

    return mask_rgb, mosaic


__all__ = ["rgb_to_bayer_mosaic"]
