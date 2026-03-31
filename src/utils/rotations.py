from __future__ import annotations

import torch


def _safe_sign(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternion(s) in wxyz format to rotation matrix/matrices.

    Args:
        quaternions: Tensor with shape (..., 4), ordered as (w, x, y, z).

    Returns:
        Tensor with shape (..., 3, 3).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(
            f"quaternion_to_matrix expects shape (..., 4), got {quaternions.shape}."
        )

    q = quaternions
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(q.dtype).eps)

    w, x, y, z = q.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    two = 2.0

    m00 = ww + xx - yy - zz
    m01 = two * (xy - wz)
    m02 = two * (xz + wy)

    m10 = two * (xy + wz)
    m11 = ww - xx + yy - zz
    m12 = two * (yz - wx)

    m20 = two * (xz - wy)
    m21 = two * (yz + wx)
    m22 = ww - xx - yy + zz

    matrix = torch.stack(
        (
            torch.stack((m00, m01, m02), dim=-1),
            torch.stack((m10, m11, m12), dim=-1),
            torch.stack((m20, m21, m22), dim=-1),
        ),
        dim=-2,
    )
    return matrix


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix/matrices to quaternion(s) in wxyz format.

    Args:
        matrix: Tensor with shape (..., 3, 3).

    Returns:
        Tensor with shape (..., 4), ordered as (w, x, y, z).
    """
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(
            f"matrix_to_quaternion expects shape (..., 3, 3), got {matrix.shape}."
        )

    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    eps = torch.finfo(matrix.dtype).eps

    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=eps))
    qx = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=eps))
    qy = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=eps))
    qz = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=eps))

    qx = qx * _safe_sign(m21 - m12)
    qy = qy * _safe_sign(m02 - m20)
    qz = qz * _safe_sign(m10 - m01)

    q = torch.stack((qw, qx, qy, qz), dim=-1)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    return q


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternion(s) in wxyz format to axis-angle vectors.

    Args:
        quaternions: Tensor with shape (..., 4), ordered as (w, x, y, z).

    Returns:
        Tensor with shape (..., 3) where magnitude is the rotation angle (rad).
    """
    if quaternions.shape[-1] != 4:
        raise ValueError(
            f"quaternion_to_axis_angle expects shape (..., 4), got {quaternions.shape}."
        )

    q = quaternions
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(q.dtype).eps)

    w = q[..., :1]
    xyz = q[..., 1:]

    # Canonicalize to w >= 0 for continuity.
    sign = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
    w = w * sign
    xyz = xyz * sign

    sin_half = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w.clamp_min(torch.finfo(q.dtype).eps))

    small = sin_half < 1e-8
    axis = xyz / sin_half.clamp_min(1e-8)
    axis = torch.where(small, torch.zeros_like(axis), axis)

    return axis * angle


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix/matrices to axis-angle vectors."""
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

__all__ = [
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "quaternion_to_axis_angle",
    "matrix_to_axis_angle",
]
