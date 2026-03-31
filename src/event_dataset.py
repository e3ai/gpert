import logging
import os
import random
from typing import Any, Callable, Optional, TypedDict, Tuple, Union, cast
import torch
from torch.cuda.amp.autocast_mode import autocast
import cv2

from .utils.motion import slerp, convert_coordinates
from .utils.rotations import matrix_to_quaternion, quaternion_to_matrix
from .types import Trajectory, Events, Intrinsics
from .third_party.threedgrut.threedgrut.datasets.protocols import Batch
from .config import (
    InterpMethod,
    DataloaderMethod,
    Config,
)


class EventBatch(TypedDict):
    x: torch.Tensor  # x coordinates of events
    y: torch.Tensor  # y coordinates of events
    t: torch.Tensor  # timestamps of events
    p: torch.Tensor  # polarities of events
    events: torch.Tensor  # (x,y,t,p) of events
    pose_start: torch.Tensor  # pose at start time (position + quaternion)
    pose_end: torch.Tensor  # pose at end time (position + quaternion)
    pose_middle: torch.Tensor  # pose at middle time (position + quaternion)
    mask: Optional[torch.Tensor]  # optional mask for events


class EventDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg: Config,
        logger: Optional[logging.Logger] = None,
        device: torch.device = torch.device("cuda"),
        train: bool = True,
    ) -> None:
        """Initialize EventDataset
        """
        self.logger = logger
        self.device = device
        device_str = str(device)
        self.train = train
        self._cuda_timer_enabled = torch.cuda.is_available() and device_str.startswith(
            "cuda"
        )
        self.batch_size = 1
        self.optimizer = None
        self.method = cfg.dataloader_method
        self.data_type = cfg.data_type
        self.interp_method = cfg.interp_method
        self.randomize_offset = cfg.randomize_offset
        self.accumulation_time = cfg.accumulation_time
        self.accumulation_num = cfg.accumulation_num
        
        events, poses, intrinsics = self.load_data_robust_e_nerf(cfg, device, logger)
        self.events = events
        self.poses = poses
        self.intrinsics = intrinsics
        self.endtime = self.events.t[-1]
        self.starttime = self.events.t[0]
        timerange = (self.endtime - self.starttime) / 1e9
        self.image_h = intrinsics.image_height
        self.image_w = intrinsics.image_width
        
        if self.method == DataloaderMethod.TIME:
            self.subinterval_num = int(timerange / self.accumulation_time)
        elif self.method == DataloaderMethod.NUM:
            self.subinterval_num = int(len(self.events.x) / self.accumulation_num)

        directions = EventDataset.__get_ray_directions(
            intrinsics=intrinsics,
            device=self.device,
            ray_jitter=None,
        )
        if isinstance(directions, tuple):
            directions = directions[0]
        self.rays_o_cam = torch.zeros(
            (self.batch_size, self.image_h, self.image_w, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.rays_d_cam = directions.reshape(
            (1, self.image_h, self.image_w, 3)
        ).contiguous()
        
        # Load mask
        mask_path = os.path.join(cfg.data_root, "mask.png")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(bool)
            mask = torch.from_numpy(mask).to(self.device)
            self.mask = mask
        else:
            self.mask = None

    def reset_intrinsics(self, intrinsics: Intrinsics):
        self.intrinsics = intrinsics
        directions = EventDataset.__get_ray_directions(
            intrinsics=intrinsics,
            device=self.device,
            ray_jitter=None,
        )
        if isinstance(directions, tuple):
            directions = directions[0]

        self.rays_o_cam = torch.zeros(
            (self.batch_size, self.image_h, self.image_w, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.rays_d_cam = directions.reshape(
            (1, self.image_h, self.image_w, 3)
        ).contiguous()

    def __len__(self):
        return self.subinterval_num - 1  # to avoid indexerror

    def _time_to_index_range(
        self, start_time: Union[torch.Tensor, int], end_time: Union[torch.Tensor, int]
    ) -> Tuple[int, int]:
        """Convert time boundaries to slice indices without allocating boolean masks."""

        t_tensor = self.events.t

        def _to_tensor(value: Union[torch.Tensor, int]) -> torch.Tensor:
            if isinstance(value, torch.Tensor):
                tensor = value.to(device=t_tensor.device, dtype=t_tensor.dtype)
            else:
                tensor = torch.tensor(
                    value, device=t_tensor.device, dtype=t_tensor.dtype
                )

            if tensor.numel() != 1:
                raise ValueError("Timestamp tensors must be scalar.")

            return tensor.reshape(1)

        start_tensor = _to_tensor(start_time)
        end_tensor = _to_tensor(end_time)

        start_idx = torch.searchsorted(t_tensor, start_tensor, right=False)
        end_idx = torch.searchsorted(t_tensor, end_tensor, right=False)

        return int(start_idx.item()), int(end_idx.item())

    def __getitem__(self, idx: int) -> EventBatch:
        """Get subinterval events

        Args:
            idx (int): index of subinterval

        Returns:
            Tuple[torch.Tensor, ...]: x, y, t, p, pose_start, pose_end
        """
        if self.method == DataloaderMethod.TIME:
            start = self.starttime + int(idx * self.accumulation_time * 1e9)
            end = self.starttime + int((idx + 1) * self.accumulation_time * 1e9)
            start_idx, end_idx = self._time_to_index_range(start, end)
            event_slice = slice(start_idx, end_idx)

            # Get pose at start/end time
            pose_start = self.get_pose_at(start, method=self.interp_method)
            pose_end = self.get_pose_at(end, method=self.interp_method)
            pose_middle = self.get_pose_at(
                torch.tensor(int((start + end) / 2), dtype=torch.int64),
                method=self.interp_method,
            )
        elif self.method == DataloaderMethod.NUM:
            start = int(self.accumulation_num * idx)
            end = start + int(self.accumulation_num)

            if self.train and self.randomize_offset:
                # Random sampling
                if idx > 0 and idx < len(self) - 1:
                    random_offset = random.randint(
                        -self.accumulation_num // 2, self.accumulation_num // 2
                    )
                    start += random_offset
                    end += random_offset

            event_slice = slice(start, end)

            # Get pose at start/end/middle time
            pose_start = self.get_pose_at(
                self.events.t[start], method=self.interp_method
            ).to(dtype=torch.float64)
            pose_end = self.get_pose_at(
                self.events.t[end], method=self.interp_method
            ).to(dtype=torch.float64)
            pose_middle = self.get_pose_at(
                self.events.t[int((start + end) / 2)], method=self.interp_method
            ).to(dtype=torch.float64)

        assert (
            pose_start.dtype == pose_end.dtype == pose_middle.dtype == torch.float64
        ), "Pose tensors must be float64"
        
        x = self.events.x[event_slice]
        y = self.events.y[event_slice]
        t = self.events.t[event_slice]
        p = self.events.p[event_slice]
        events = torch.stack((y, x, t, p), dim=-1)  # (N, 4), keep legacy order (y,x,t,p)

        event_batch_dict = {
            "events": events,  # (x,y,t,p)
            "pose_start": pose_start,
            "pose_end": pose_end,
            "pose_middle": pose_middle,
        }

        if self.mask is not None:
            event_batch_dict["mask"] = self.mask

        event_batch = EventBatch(**event_batch_dict)
        return event_batch

    def get_item_by_timestamp(
        self, t: torch.Tensor, t_prev: torch.Tensor
    ) -> EventBatch:
        """Get events and poses from timestamps (for evaluation)"""
        start_idx, end_idx = self._time_to_index_range(t_prev, t)
        event_slice = slice(start_idx, end_idx)

        # Get pose at start/end time
        pose_start = self.get_pose_at(t_prev, method=self.interp_method)
        pose_end = self.get_pose_at(t, method=self.interp_method)
        pose_middle = self.get_pose_at((t + t_prev) / 2, method=self.interp_method)
        
        x = self.events.x[event_slice]
        y = self.events.y[event_slice]
        t = self.events.t[event_slice]
        p = self.events.p[event_slice]
        events = torch.stack((y, x, t, p), dim=-1)  # (N, 4), keep legacy order (y,x,t,p)

        event_batch_dict = {
            "events": events,  # (x,y,t,p)
            "p": self.events.p[event_slice],  # p
            "pose_start": pose_start,
            "pose_end": pose_end,
            "pose_middle": pose_middle,
        }

        if self.mask is not None:
            event_batch_dict["mask"] = self.mask

        event_batch = EventBatch(**event_batch_dict)
        return event_batch

    def get_pose_at(
        self, t: torch.Tensor, method: InterpMethod = InterpMethod.LIN
    ) -> torch.Tensor:
        return self.poses.get_pose_at(query_t=t, method=method)

    def get_image_batch_with_intrinsics(self, T_to_world: torch.Tensor) -> Batch:
        """Add the intrinsics to the batch (the output of __getitem__).

        Args:
            pose (torch.Tensor): pose of the camera (pos + quaternion)
                shape: (B, 7)
        """
        sample = {
            "rays_ori": self.rays_o_cam,
            "rays_dir": self.rays_d_cam,
            "T_to_world": T_to_world,
            "intrinsics": [self.intrinsics.fx, self.intrinsics.fy, self.intrinsics.cx, self.intrinsics.cy],
        }

        return Batch(**sample)

    def get_event_batch_with_intrinsics(
        self,
        events: torch.Tensor,
        T_start: torch.Tensor,
        T_end: torch.Tensor,
        method: InterpMethod = InterpMethod.SLERP,
    ) -> Batch:
        """Create a batch with rays and intrinsics for each events.

        Args:
            event (torch.Tensor): events, shape: (N, 4) where N is the number of events (x,y,t,p)
            T_start (torch.Tensor): transformation matrix at start time, shape: (4, 4)
            T_end (torch.Tensor): transformation matrix at end time, shape: (4, 4)
            **t is world coordinate, and R is c2w**

        Returns:
            dict:
                rays_ori (torch.Tensor): origin of rays in camera coordinate, shape: (N, 3)
                rays_dir (torch.Tensor): direction of rays in camera coordinate, shape: (N, 3)
                T_to_world (torch.Tensor): c2w transformation matrix  for each event, shape: (4, 4)
                intrinsics (list): camera intrinsics (fx, fy, cx, cy)
        """
        # Get timestamps
        t = events[:, 2]  # timestamps
        N = len(t)  # number of events

        # Get time range for interpolation
        t1 = t[0]
        t2 = t[-1]

        # Calculate interpolation weights for all events at once
        alpha = (t - t1) / (t2 - t1)  # (N,)

        # Extract rotation matrices and translation vectors from transformation matrices
        R_start = T_start[:3, :3]  # (3, 3), c2w
        t_start = T_start[:3, 3]  # (3,), w
        R_end = T_end[:3, :3]  # (3, 3)
        t_end = T_end[:3, 3]  # (3,)

        t_start_c = -R_start.T @ t_start  # w -> c
        # t_end = -R_end.T @ t_end  # w -> c

        # Batch interpolation for all events
        if method == InterpMethod.LIN:
            # Linear interpolation for rotation matrices and translation vectors
            alpha_expanded = alpha.unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1) for rotation
            alpha_t = alpha.unsqueeze(-1)  # (N, 1) for translation

            R_interp = (
                R_start.unsqueeze(0) * (1 - alpha_expanded)
                + R_end.unsqueeze(0) * alpha_expanded
            )  # (N, 3, 3)
            t_interp = (
                t_start.unsqueeze(0) * (1 - alpha_t) + t_end.unsqueeze(0) * alpha_t
            )  # (N, 3)

        elif method == InterpMethod.SLERP:
            # Convert rotation matrices to quaternions for slerp
            quat_start = matrix_to_quaternion(R_start.unsqueeze(0))  # (1, 4(w,x,y,z))
            quat_end = matrix_to_quaternion(R_end.unsqueeze(0))  # (1, 4)

            # Vectorized linear interpolation for translation
            alpha_t = alpha.unsqueeze(-1)  # (N, 1)
            t_interp = (
                t_start.unsqueeze(0) * (1 - alpha_t) + t_end.unsqueeze(0) * alpha_t
            )  # (N, 3), world

            # Vectorized spherical linear interpolation for quaternion
            quat_start = quat_start.to(dtype=torch.float32)
            quat_end = quat_end.to(dtype=torch.float32)
            alpha = alpha.to(dtype=torch.float32)
            quat_interp = slerp(
                quat_start.expand(N, -1), quat_end.expand(N, -1), alpha.unsqueeze(-1)
            )  # (N, 4)
            R_interp = quaternion_to_matrix(quat_interp)  # (N, 3, 3), c2w

        else:
            raise ValueError(
                f"Unknown interpolation method: {method}. Use 'lin' or 'slerp'."
            )

        # Get ray origins for each event (in start camera coordinate)
        rays_ori = R_start.unsqueeze(0).transpose(1, 2) @ t_interp.unsqueeze(
            -1
        )  # (N, 3, 1)
        rays_ori = rays_ori + t_start_c.unsqueeze(0).unsqueeze(-1)  # (N, 3, 1)
        rays_ori = rays_ori.squeeze(-1)  # (N, 3)

        # Get ray directions for each event
        event_in_camera_coord = EventDataset.__get_event_in_camera_coordinate(
            events,
            intrinsics=self.intrinsics,
        ).reshape((N, 3))

        # Transform rays_dir to camera_start coordinate system
        event_in_camera_coord = event_in_camera_coord.to(R_interp)
        event_in_world_coord = R_interp @ event_in_camera_coord.unsqueeze(
            -1
        ) + t_interp.unsqueeze(-1)  # (N, 3, 1)
        event_in_camera_start_coord = (
            R_start.unsqueeze(0).transpose(1, 2) @ event_in_world_coord
        )  # (N, 3, 1)
        event_in_camera_start_coord = event_in_camera_start_coord + t_start_c.unsqueeze(
            0
        ).unsqueeze(-1)  # (N, 3, 1)
        rays_dir = event_in_camera_start_coord.squeeze() - rays_ori  # (N, 3)
        rays_dir = torch.nn.functional.normalize(
            rays_dir, dim=-1
        )  # Normalize directions

        # Unsqueeze
        rays_ori = rays_ori.unsqueeze(0).contiguous()
        rays_dir = rays_dir.unsqueeze(0).contiguous()
        T_start = T_start.unsqueeze(0)  # (1, 4, 4)

        # gut input must be float32
        rays_ori = rays_ori.to(dtype=torch.float32)
        rays_dir = rays_dir.to(dtype=torch.float32)
        T_start = T_start.to(dtype=torch.float32)

        sample = {
            "rays_ori": rays_ori,
            "rays_dir": rays_dir,
            "T_to_world": T_start,  # t is world coordinate
            "intrinsics": [self.intrinsics.fx, self.intrinsics.fy, self.intrinsics.cx, self.intrinsics.cy],
        }

        return Batch(**sample)

    def set_eval(self):
        self.accumulation_num = 204800
        self.train = False
        self.subinterval_num = int(len(self.events.x) / self.accumulation_num)
        self.accumulation_num_coeff = 1

    @staticmethod
    @autocast(dtype=torch.float32)
    def __get_ray_directions(
        intrinsics: Intrinsics, device=torch.device("cpu"), ray_jitter=None, return_uv=False, flatten=True
    ):
        """
        Get ray directions for all pixels in camera coordinate [right down front].
        Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
                ray-tracing-generating-camera-rays/standard-coordinate-systems

        Inputs:
            H, W: image height and width
            fx, fy, cx, cy: camera intrinsics
            ray_jitter: Optional RayJitter component, for whether the ray passes randomly inside the pixel
            return_uv: whether to return uv image coordinates

        Outputs: (shape depends on @flatten)
            directions: (H, W, 3) or (H*W, 3), the direction of the rays in camera coordinate
            uv: (H, W, 2) or (H*W, 2) image coordinates
        """
        dtype = torch.float32
        y_coords = torch.arange(intrinsics.image_height, device=device, dtype=dtype)
        x_coords = torch.arange(intrinsics.image_width, device=device, dtype=dtype)
        v, u = torch.meshgrid(y_coords, x_coords, indexing="ij")
        grid = torch.stack((u, v), dim=-1)  # (H, W, 2), [x, y]

        if ray_jitter is None:  # pass by the center
            directions = torch.stack(
                [(u - intrinsics.cx + 0.5) / intrinsics.fx, (v - intrinsics.cy + 0.5) / intrinsics.fy, torch.ones_like(u)], -1
            )
        else:
            jitter = ray_jitter(u.shape)
            directions = torch.stack(
                [
                    ((u + jitter[:, :, 0]) - intrinsics.cx) / intrinsics.fx,
                    ((v + jitter[:, :, 1]) - intrinsics.cy) / intrinsics.fy,
                    torch.ones_like(u),
                ],
                -1,
            )
        if flatten:
            directions = directions.reshape(-1, 3)
            grid = grid.reshape(-1, 2)

        if return_uv:
            return directions, grid

        return torch.nn.functional.normalize(directions, dim=-1)

    @staticmethod
    @autocast(dtype=torch.float32)
    def __get_event_in_camera_coordinate(
        events, intrinsics: Intrinsics
    ):
        """
        Get ray directions for all pixels in camera coordinate [right down front].
        Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
                ray-tracing-generating-camera-rays/standard-coordinate-systems

        Inputs:
            events: (N, 4) events (x,y,t,p)
            intrinsics: Intrinsics object containing camera parameters

        Outputs: (shape depends on @flatten)
            directions: (N, 3), the direction of the rays in camera coordinate
        """

        fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
        u = events[:, 0]  # x coordinates
        v = events[:, 1]  # y coordinates
        
        event_in_camera_coordinate = torch.stack(
            [(u - cx + 0.5) / fx, (v - cy + 0.5) / fy, torch.ones_like(u)], -1
        )

        # return torch.nn.functional.normalize(directions, dim=-1)
        return event_in_camera_coordinate

    def load_data_robust_e_nerf(
        self, cfg: Config, device: torch.device, logger: Optional[logging.Logger]
    ) -> Tuple[Events, Trajectory, Intrinsics]:
        """Load Events/Trajectory/Intrinsics from a Robust e-NeRF style dataset.

        Expected files under `cfg.data_root`:
        - raw_events.npz: keys {position(N,2 int), timestamp(N), polarity(N)}
        - camera_poses.npz: keys {T_wc_position(N,3), T_wc_orientation(N,3,3), T_wc_timestamp(N)}
        - camera_calibration.npz: keys {img_height, img_width, intrinsics(3,3), distortion_model, distortion_params, bayer_pattern}

        Returns (Events, Trajectory, Intrinsics).
        """
        import numpy as np

        root = cfg.data_root
        assert os.path.isdir(root), f"Data root directory does not exist: {root}"

        # Load raw events
        undistorted_events_path = os.path.join(root, cfg.event_name)
        raw_events_path = os.path.join(root, "raw_events.npz")

        if os.path.isfile(undistorted_events_path):
            raw_events = np.load(undistorted_events_path)
            if logger:
                logger.info(f"Loaded undistorted events from {undistorted_events_path}")
        elif os.path.isfile(raw_events_path):
            raw_events = np.load(raw_events_path)
            if logger:
                logger.info(f"Loaded raw events from {raw_events_path}")
        else:
            raise FileNotFoundError(
                f"Neither undistorted_events.npz nor raw_events.npz found in {root}"
            )

        pos = raw_events.get("position")  # (N,2) [x,y]
        ts = raw_events.get("timestamp")  # (N,)
        pol = raw_events.get("polarity")  # (N,)
        assert pos is not None and ts is not None and pol is not None, (
            "raw_events.npz missing required keys"
        )

        x = torch.from_numpy(pos[:, 0].astype(np.int64)).to(device)
        y = torch.from_numpy(pos[:, 1].astype(np.int64)).to(device)
        t = torch.from_numpy(ts.astype(np.int64)).to(device)  # Cast timestamps to int64 (ns)
        p = torch.from_numpy(pol).to(device)
        events = Events(t=t, x=x, y=y, p=p)

        # Load camera poses
        cam_poses_path = os.path.join(root, cfg.traj_name)
        assert os.path.isfile(cam_poses_path), (
            f"camera_poses.npz not found at {cam_poses_path}"
        )
        cam_poses = np.load(cam_poses_path)

        T_wc_position = cam_poses["T_wc_position"]  # (N,3)
        T_wc_orientation = cam_poses["T_wc_orientation"]  # (N,4) xyzw quat
        T_wc_timestamp = cam_poses["T_wc_timestamp"]  # (N,)

        t_pose = torch.from_numpy(T_wc_timestamp.astype(np.int64)).to(device)
        pos_w = torch.from_numpy(T_wc_position).to(device)
        q_xyzw = torch.from_numpy(T_wc_orientation).to(device)
        q_wxyz = q_xyzw[:, [3, 0, 1, 2]]  # (N,4) xyzw -> wxyz

        # Convert coordinates once at load time to avoid repeated per-query conversion.
        R_c2w = quaternion_to_matrix(q_wxyz)
        R_c2w, pos_w = convert_coordinates(
            R_c2w,
            pos_w,
            cfg.data_type,
            device=device,
            precision=torch.float64,
        )
        q_wxyz = matrix_to_quaternion(R_c2w)

        trajectory = Trajectory(t=t_pose, position=pos_w, orientation=q_wxyz)

        # Load camera calibration and build Intrinsics
        calib_path = os.path.join(root, cfg.camera_info_name)
        assert os.path.isfile(calib_path), (
            f"camera_calibration not found at {calib_path}"
        )
        calib = np.load(calib_path)

        image_height = (
            int(calib["img_height"])
            if "img_height" in calib
            else int(calib["image_height"])
        )  # type: ignore
        image_width = (
            int(calib["img_width"])
            if "img_width" in calib
            else int(calib["image_width"])
        )  # type: ignore
        K = torch.from_numpy(calib["intrinsics"].astype(np.float64)).to(device)  # (3,3)

        camera_info = Intrinsics(
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            image_width=image_width,
            image_height=image_height,
        )

        logger.info(
            f"Loaded Robust e-NeRF dataset: events={len(t)}, poses={len(t_pose)}, size=({image_height},{image_width})"
        )

        return events, trajectory, camera_info
