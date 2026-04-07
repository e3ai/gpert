from typing import Tuple, Optional

import torch


def motion_field_optimised_torch(
    v: torch.Tensor,
    w: torch.Tensor,
    z: torch.Tensor,
    f: float,
    image_shape: Tuple[int, int],
    device: str,
) -> torch.Tensor:
    """
    Modified to handle batched inputs for v, w, and z.
    :param v: linear velocity, shape is (B, 3)
    :param w: angular velocity, shape is (B, 3)
    :param z: inverse depth of the scene, shape is (B, H, W)
    :returns: flow of shape (B, 2{y, x}, H, W)
    """
    # Define image coordinate which is centered at the image center
    hh = image_shape[0] / 2
    hw = image_shape[1] / 2
    heights = torch.linspace(-hh, hh, z.shape[1]).to(device)
    widths = torch.linspace(-hw, hw, z.shape[2]).to(device)
    f = torch.tensor(f).to(device)

    v = v.to(device)
    w = w.to(device)
    z = z.to(device)

    pyr_v = v
    pyr_w = w

    # Create meshgrid
    y, x = torch.meshgrid(heights, widths, indexing="ij")
    y = y.to(device)
    x = x.to(device)
    y = y.unsqueeze(0).expand(z.shape[0], -1, -1)  # Expand for batch dimension
    x = x.unsqueeze(0).expand(z.shape[0], -1, -1)

    # Compute flow (translation + rotation)
    flow_x_trans = 1.0 / z * (-f * pyr_v[:, 0:1, None] + x * pyr_v[:, 2:3, None])
    flow_y_trans = 1.0 / z * (-f * pyr_v[:, 1:2, None] + y * pyr_v[:, 2:3, None])
    flow_x_rot = (
        1.0
        / f
        * (
            x * y * pyr_w[:, 0:1, None]
            - (torch.square(f) + torch.square(x)) * pyr_w[:, 1:2, None]
            + f * y * pyr_w[:, 2:3, None]
        )
    )
    flow_y_rot = (
        1.0
        / f
        * (
            (torch.square(f) + torch.square(y)) * pyr_w[:, 0:1, None]
            - x * y * pyr_w[:, 1:2, None]
            - f * x * pyr_w[:, 2:3, None]
        )
    )
    flow_x = flow_x_trans + flow_x_rot
    flow_y = flow_y_trans + flow_y_rot
    flow = torch.stack([flow_y, flow_x], dim=1)  # Shape: (B, 2, H, W)
    return flow


def compute_motion_field(
    v: torch.Tensor,
    w: torch.Tensor,
    z: torch.Tensor,
    f: torch.Tensor,
    image_size: Tuple[int, int],
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Modified to handle batched inputs for v, w, and z.
    :param v: linear velocity, shape is (B, 3)
    :param w: angular velocity, shape is (B, 3)
    :param z: depth, shape is (B, H, W)  # not inverse depth?
    :param f: focal length (float)
    :returns: motion flow of shape (B, 2{y,x}, H, W)
    """
    if torch.is_tensor(v) and torch.is_tensor(w) and torch.is_tensor(z):
        # Ensure v and w have batch dimensions
        if v.dim() == 1:
            v = v.unsqueeze(0)  # Shape: (B, 3)
        if w.dim() == 1:
            w = w.unsqueeze(0)  # Shape: (B, 3)
        motion_field = motion_field_optimised_torch(v, w, z, f, image_size, str(device))
    else:
        raise NotImplementedError

    return motion_field
