from typing import Tuple

import torch
from torch import Tensor, Size, lerp, norm, zeros_like

from ..config import DataType
from .rotations import matrix_to_axis_angle, quaternion_to_matrix


def slerp(v0: Tensor, v1: Tensor, t: float | Tensor, DOT_THRESHOLD=0.9995):
    """Spherical linear interpolation."""
    if v0.shape[-1] != v1.shape[-1]:
        raise AssertionError("last dimension of v0 and v1 must match")

    if v0.device != v1.device:
        raise ValueError("v0 and v1 must reside on the same device")

    common_dtype = torch.promote_types(v0.dtype, v1.dtype)
    v0 = v0.to(dtype=common_dtype)
    v1 = v1.to(dtype=common_dtype)

    v0, v1 = torch.broadcast_tensors(v0, v1)

    if not isinstance(t, Tensor):
        t = v0.new_tensor(t)
    else:
        t = t.to(device=v0.device, dtype=common_dtype)

    while t.dim() < v0.dim():
        t = t.unsqueeze(-1)

    v0_norm = norm(v0, dim=-1)
    v1_norm = norm(v1, dim=-1)

    v0_normed = v0 / v0_norm.unsqueeze(-1)
    v1_normed = v1 / v1_norm.unsqueeze(-1)

    dot = (v0_normed * v1_normed).sum(-1)

    flip_mask = dot < 0
    if flip_mask.any():
        v1 = v1.where(~flip_mask.unsqueeze(-1), -v1)
        v1_normed = v1_normed.where(~flip_mask.unsqueeze(-1), -v1_normed)
        dot = dot.abs()

    dot_mag = dot.abs()
    gotta_lerp = dot_mag.isnan() | (dot_mag > DOT_THRESHOLD)
    can_slerp = ~gotta_lerp

    t_batch_dim_count: int = max(0, t.dim() - v0.dim()) if isinstance(t, Tensor) else 0
    t_batch_dims: Size = (
        t.shape[:t_batch_dim_count] if isinstance(t, Tensor) else Size([])
    )
    out = zeros_like(v0.expand(*t_batch_dims, *[-1] * v0.dim()))

    if gotta_lerp.any():
        lerped = lerp(v0, v1, t)
        out = lerped.where(gotta_lerp.unsqueeze(-1), out)

    if can_slerp.any():
        theta_0 = dot.arccos().unsqueeze(-1)
        sin_theta_0 = theta_0.sin()
        theta_t = theta_0 * t
        sin_theta_t = theta_t.sin()
        s0 = (theta_0 - theta_t).sin() / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        slerped = s0 * v0 + s1 * v1

        out = slerped.where(can_slerp.unsqueeze(-1), out)

    out_norm = norm(out, dim=-1, keepdim=True)
    eps = torch.finfo(out.dtype).eps
    out = out / out_norm.clamp_min(eps)

    return out


def convert_coordinates(
    R: torch.Tensor,
    T: torch.Tensor,
    data_type: DataType,
    device: torch.device,
    precision: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert dataset/world coordinates into renderer coordinates."""
    N = R.shape[0]

    R = R.to(dtype=precision)
    T = T.to(dtype=precision)

    if data_type == DataType.ROBUST_E_NERF:
        return R, T

    if data_type == DataType.ROBUST_E_NERF_TEST:
        S = torch.diag(torch.tensor([-1.0, 1.0, 1.0], device=device, dtype=precision))
        S = S.unsqueeze(0).expand(N, -1, -1)
        R = torch.bmm(torch.bmm(S, R), S)
        T = torch.bmm(S, T.unsqueeze(-1)).squeeze(-1)

        transform_matrix = torch.tensor(
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
            device=device,
            dtype=precision,
        ).repeat(N, 1, 1)
        R = torch.bmm(transform_matrix, R)
        T = torch.bmm(transform_matrix, T.unsqueeze(-1)).squeeze(-1)
        return R, T

    raise NotImplementedError(f"Unknown data type: {data_type}")


def compute_velocity_and_angular_velocity(
    pose_start: torch.Tensor,
    pose_end: torch.Tensor,
    pose_middle: torch.Tensor,
    t: torch.Tensor,
    precision: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute linear and angular velocity in the mid-camera coordinate frame."""
    t_sec = t.clone().to(precision)
    t_sec = ((t_sec - t_sec[:, 0]) / 1e9).to(precision)
    delta_t = t_sec[:, -1:] - t_sec[:, :1]

    rotmat_middle = quaternion_to_matrix(pose_middle[:, 3:7]).to(dtype=pose_start.dtype)
    rotmat_start = quaternion_to_matrix(pose_start[:, 3:7]).to(dtype=pose_start.dtype)
    rotmat_end = quaternion_to_matrix(pose_end[:, 3:7]).to(dtype=pose_start.dtype)

    position_start = torch.bmm(
        rotmat_middle.transpose(1, 2), pose_start[:, :3].unsqueeze(-1)
    ).squeeze(-1) - torch.bmm(
        rotmat_middle.transpose(1, 2), pose_middle[:, :3].unsqueeze(-1)
    ).squeeze(-1)
    position_end = torch.bmm(
        rotmat_middle.transpose(1, 2), pose_end[:, :3].unsqueeze(-1)
    ).squeeze(-1) - torch.bmm(
        rotmat_middle.transpose(1, 2), pose_middle[:, :3].unsqueeze(-1)
    ).squeeze(-1)
    velocity = (position_end - position_start) / delta_t

    angular_velocity = rotmat_angular_velocity(
        rotmat_start, rotmat_end, rotmat_middle, delta_t
    )

    return velocity, angular_velocity


def rotmat_angular_velocity(R_start, R_end, R_middle, delta_t):
    R_rel = torch.bmm(R_end, R_start.transpose(1, 2))
    R_rel_middle = torch.bmm(R_middle.transpose(1, 2), R_rel)
    R_rel_middle = torch.bmm(R_rel_middle, R_middle)

    rotvec_rel = matrix_to_axis_angle(R_rel_middle)

    angular_velocity = rotvec_rel / delta_t
    return angular_velocity


__all__ = [
    "slerp",
    "convert_coordinates",
    "compute_velocity_and_angular_velocity",
    "rotmat_angular_velocity",
]
