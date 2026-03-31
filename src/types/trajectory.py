from typing import NewType
import torch
from ..utils import slerp
from ..config import InterpMethod

# QuaternionWXYZ = Annotated[torch.Tensor, "Quaternion in w,x,y,z format"]
QuaternionWXYZ = NewType("QuaternionWXYZ", torch.Tensor)


class Trajectory:
    """Represents a trajectory in 3D space.

    Attributes:
        t: A tensor of timestamps.
        position: A tensor of positions in 3D space, shape (N, 3).
        orientation: A tensor of orientations represented as quaternions, shape (N, 4(wxyz)).
    """

    def __init__(
        self,
        t: torch.Tensor,
        position: torch.Tensor,
        orientation: torch.Tensor,
        orientation_is_xyzw: bool = False,
    ):
        if t.ndim != 1:
            raise ValueError(f"t must be 1-D, got shape {t.shape}.")
        if t.numel() < 2:
            raise ValueError("Trajectory requires at least 2 timestamp samples.")
        if not torch.all(t[1:] >= t[:-1]):
            raise ValueError("t must be sorted in non-decreasing order.")
        if position.shape != (t.shape[0], 3):
            raise ValueError(
                f"position must have shape ({t.shape[0]}, 3), got {position.shape}."
            )
        if orientation.shape != (t.shape[0], 4):
            raise ValueError(
                f"orientation must have shape ({t.shape[0]}, 4), got {orientation.shape}."
            )

        self.t = t.contiguous()
        self.position = position
        self.orientation = orientation

        self.bin_width = torch.diff(self.t)
        self.orientation_is_xyzw = orientation_is_xyzw

    def quaternion_to_wxyz(self, is_xyzw: bool) -> QuaternionWXYZ:
        """Convert quaternion from xyzw to wxyz format."""
        if is_xyzw:
            return QuaternionWXYZ(
                torch.stack(
                    (
                        self.orientation[:, 3],
                        self.orientation[:, 0],
                        self.orientation[:, 1],
                        self.orientation[:, 2],
                    ),
                    dim=1,
                )
            )
        else:
            return QuaternionWXYZ(self.orientation)


    def get_pose_at(
        self,
        query_t: torch.Tensor | int,
        method: InterpMethod = InterpMethod.LIN,
    ) -> torch.Tensor:
        """Interpolate poses at timestamp(s) and return pose vector(s) [x, y, z, qw, qx, qy, qz]."""
        if isinstance(query_t, torch.Tensor):
            t_query = query_t.to(device=self.t.device, dtype=self.t.dtype)
        else:
            t_query = torch.tensor(query_t, device=self.t.device, dtype=self.t.dtype)

        if t_query.dim() > 1:
            raise ValueError(f"query_t must be scalar or 1-D, got shape {t_query.shape}.")

        is_scalar = t_query.dim() == 0
        if is_scalar:
            t_query = t_query.unsqueeze(0)

        t_query = t_query.contiguous()
        orientation_wxyz = self.quaternion_to_wxyz(self.orientation_is_xyzw)
        insertion = torch.searchsorted(self.t, t_query, right=False)

        first_pose = torch.cat([self.position[0], orientation_wxyz[0]], dim=0)
        last_pose = torch.cat([self.position[-1], orientation_wxyz[-1]], dim=0)

        pose = torch.empty(
            (t_query.shape[0], 7),
            device=self.position.device,
            dtype=self.position.dtype,
        )

        at_start = insertion == 0
        at_end = insertion >= len(self.t)
        interior = ~(at_start | at_end)

        if at_start.any():
            pose[at_start] = first_pose
        if at_end.any():
            pose[at_end] = last_pose

        if interior.any():
            idx2 = insertion[interior]
            idx1 = idx2 - 1

            t1 = self.t[idx1]
            t2 = self.t[idx2]
            alpha = (t_query[interior] - t1) / (t2 - t1)

            pos_1 = self.position[idx1]
            pos_2 = self.position[idx2]
            pos = pos_1 * (1 - alpha.unsqueeze(-1)) + pos_2 * alpha.unsqueeze(-1)

            if method == InterpMethod.LIN:
                quat_1 = orientation_wxyz[idx1]
                quat_2 = orientation_wxyz[idx2]
                quat = quat_1 * (1 - alpha.unsqueeze(-1)) + quat_2 * alpha.unsqueeze(-1)
            elif method == InterpMethod.SLERP:
                quat_1 = orientation_wxyz[idx1].to(dtype=torch.float64)
                quat_2 = orientation_wxyz[idx2].to(dtype=torch.float64)
                quat = slerp(quat_1, quat_2, alpha.to(dtype=torch.float64))
                quat = quat.to(dtype=pos.dtype)
            else:
                raise ValueError(
                    f"Unknown interpolation method: {method}. Use 'lin' or 'slerp'."
                )

            pose[interior] = torch.cat([pos, quat], dim=1)

        if is_scalar:
            return pose[0]
        return pose
