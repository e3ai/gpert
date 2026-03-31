from __future__ import annotations
import logging
from typing import Optional, Tuple, Union
import torch

from .event_utils import compute_motion_field, linlog
from .utils.filters import motion_fusion_optimized, get_cached_diff_filter
from .event_dataset import EventDataset
from .config import BackgroundColor, DiffMethod, LogMethod, Config
from .utils.rotations import quaternion_to_matrix
from .third_party.threedgrut.threedgrut.model.model import MixtureOfGaussians
from .types import Intrinsics


class Renderer:
    def __init__(
        self,
        *,
        cfg: Config,
        device: torch.device,
        precision: torch.dtype,
        dataset: EventDataset,
        model: MixtureOfGaussians,
        logger: logging.Logger,
        background_color: BackgroundColor,
        intrinsics: Intrinsics,
        c: torch.Tensor,
        global_step: int,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.precision = precision
        self.dataset = dataset
        self.model = model
        self.logger = logger
        self.background_color = background_color
        self.intrinsics = intrinsics
        self.c = c
        self.global_step = global_step
        self.gpu_batch: Optional[dict] = None
        self._cached_diff_filter: Optional[torch.nn.Module] = None
        
    def reset_intrinsics(self, intrinsics: Intrinsics):
        self.intrinsics = intrinsics
        
    def get_diff_filter(
        self, inchannels: int = 1, precision: str = "32"
    ) -> torch.nn.Module:
        if self._cached_diff_filter is None:
            self._cached_diff_filter = get_cached_diff_filter(
                in_channels=inchannels,
                cuda_available=True,
                device=self.device,
                precision=precision,
                filter_type=self.cfg.filter_type,
            )
        return self._cached_diff_filter

    def simulate_events(
        self,
        i1: torch.Tensor,
        i2: torch.Tensor,
        c: torch.Tensor = torch.tensor(1.0),
        log_method: LogMethod = LogMethod.LOG,
        eps: float = 0.00196078431,
    ) -> torch.Tensor:
        """Simulate events from two rendered images, using log/linlog."""
        if log_method == LogMethod.LOG:
            diff = torch.log(i2 + eps) - torch.log(i1 + eps)
        elif log_method == LogMethod.LINLOG:
            diff = linlog(i2, 0.078) - linlog(i1, 0.078)
        else:
            raise NotImplementedError

        diff = diff / c

        return diff

    def render_image(
        self,
        pose: torch.Tensor,
        render_depth: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Render image from 3D Gaussian parameters and pose."""
        batch_size = pose.shape[0]
        r_c2w = quaternion_to_matrix(pose[:, 3:7])
        t_w = pose[:, :3]

        t_to_world = (
            torch.eye(4, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        )
        t_to_world[:, :3, :3] = r_c2w
        t_to_world[:, :3, 3] = t_w

        gpu_batch = self.dataset.get_image_batch_with_intrinsics(t_to_world)
        self.gpu_batch = gpu_batch
        outputs = self.model(gpu_batch, train=True)

        rendered_img = outputs["pred_rgb"].to(self.precision)

        if self.background_color is BackgroundColor.GRAY:
            bg_color = torch.ones((3,), device=self.device, dtype=self.precision) * 0.5
            rendered_img = rendered_img + bg_color * (1.0 - outputs["pred_opacity"])
        elif self.background_color is BackgroundColor.WHITE:
            bg_color = torch.ones((3,), device=self.device, dtype=self.precision)
            rendered_img = rendered_img + bg_color * (1.0 - outputs["pred_opacity"])

        if render_depth:
            rendered_depth = outputs["pred_dist"][..., 0].unsqueeze(-1)
            rendered_alpha = outputs["pred_opacity"]
            return rendered_img, rendered_depth, rendered_alpha

        return rendered_img

    def render_depth(
        self,
        events: torch.Tensor,
        pose_start: torch.Tensor,
        pose_end: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render depth of each event from 3D Gaussian parameters and pose."""
        rs_c2w = quaternion_to_matrix(pose_start[:, 3:7])
        ts_w = pose_start[:, :3]

        re_c2w = quaternion_to_matrix(pose_end[:, 3:7])
        te_w = pose_end[:, :3]

        t_to_world_s = torch.eye(4, device=self.device, dtype=self.precision)
        t_to_world_s[:3, :3] = rs_c2w
        t_to_world_s[:3, 3] = ts_w
        t_to_world_e = torch.eye(4, device=self.device, dtype=self.precision)
        t_to_world_e[:3, :3] = re_c2w
        t_to_world_e[:3, 3] = te_w

        if events.ndim == 3:
            events = events[0]

        events = events[:, [1, 0, 2, 3]]

        gpu_batch = self.dataset.get_event_batch_with_intrinsics(
            events,
            t_to_world_s,
            t_to_world_e,
            method=self.cfg.interp_method,
        )
        outputs = self.model(gpu_batch, train=True, render_type="depth", depth_grad=self.cfg.depth_grad)

        rendered_depth = outputs["pred_dist"].to(self.precision)[..., 0].unsqueeze(-1)
        rendered_opacity = outputs["pred_opacity"].to(self.precision)

        rendered_depth_adjusted = rendered_depth / (rendered_opacity + 1e-8)
        depthmap = torch.zeros(
            (self.intrinsics.image_height, self.intrinsics.image_width), device=self.device, dtype=self.precision
        )

        return rendered_depth, rendered_depth_adjusted, depthmap

    def create_diff_image(
        self,
        pose_start: torch.Tensor,
        pose_end: torch.Tensor,
        pose_middle: torch.Tensor,
        t: torch.Tensor,
        diff_method: DiffMethod,
        velocity: Optional[torch.Tensor] = None,
        angular_velocity: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Create diff image from rendered images.
        - TWICE: render two images at start and end poses, then compute diff.
        - ONCE: render one image at middle pose, then compute diff using motion field.
        """
        depth_middle = None

        if diff_method == DiffMethod.TWICE:
            img_start = self.render_image(pose_start, render_depth=False)
            img_end = self.render_image(pose_end, render_depth=False)

            assert isinstance(img_start, torch.Tensor) and isinstance(
                img_end, torch.Tensor
            )

            if self.cfg.is_color:
                pass
            else:
                img_start = (
                    img_start[..., 0] * 0.2990
                    + img_start[..., 1] * 0.5870
                    + img_start[..., 2] * 0.1140
                )
                img_end = (
                    img_end[..., 0] * 0.2990
                    + img_end[..., 1] * 0.5870
                    + img_end[..., 2] * 0.1140
                )

            diff_img = self.simulate_events(
                img_start,
                img_end,
                c=self.c,
                log_method=self.cfg.log_method,
                eps=self.cfg.log_eps,
            )
            self.logger.info(f"diff_img_max: {diff_img.max().item():.4f}")

            depth_middle = torch.zeros(
                (pose_middle.shape[0], self.intrinsics.image_height, self.intrinsics.image_width, 1),
                device=self.device,
                dtype=self.precision,
            )
            depth_middle_raw = depth_middle

        elif diff_method == DiffMethod.ONCE:
            t_nsec = t.clone().to(torch.int64)
            t_sec = (t_nsec - t_nsec[0, 0]).to(torch.float64) / 1e9
            t_sec = t_sec.to(self.precision)
            delta_t = t_sec[:, -1:] - t_sec[:, :1]
            img_middle, depth_middle_raw, alpha_middle = self.render_image(
                pose_middle, render_depth=True
            )
            alpha_middle = alpha_middle.detach()
            depth_middle = depth_middle_raw / (alpha_middle + 1e-8)

            depth_middle = depth_middle.detach()

            if self.cfg.is_color:
                pass
            else:
                img_middle = (
                    img_middle[..., 0] * 0.2990
                    + img_middle[..., 1] * 0.5870
                    + img_middle[..., 2] * 0.1140
                )

            img_start = img_middle

            if self.cfg.log_method == LogMethod.LINLOG:
                img_middle = linlog(
                    img_middle, linlog_threshold=self.cfg.linlog_threshold
                )
            elif self.cfg.log_method == LogMethod.LOG:
                img_middle = torch.log(img_middle + self.cfg.log_eps)

            clipped_depth = depth_middle.squeeze(-1)

            flow = compute_motion_field(
                v=velocity,
                w=angular_velocity,
                z=clipped_depth,
                f=self.intrinsics.fx,
                image_size=(self.intrinsics.image_height, self.intrinsics.image_width),
            )
            flow_torch = flow.to(self.device)

            mask = depth_middle == 0
            mask = mask.permute(0, 3, 1, 2).repeat(1, 2, 1, 1)
            flow_torch = flow_torch.masked_fill(mask, 0.0)

            flow_torch = flow_torch[:, [1, 0]]

            if self.cfg.is_color:
                diff_filter = self.get_diff_filter(inchannels=3, precision="64")
                img_middle = img_middle.permute(0, 3, 1, 2)
                gradient_img = diff_filter(img_middle)
                # gradient_img = gradient_img / 2

                gradient_x = gradient_img[:, [0, 2, 4], ...]
                gradient_y = gradient_img[:, [1, 3, 5], ...]
                gradient_img = torch.stack([gradient_x, gradient_y], dim=1)

                flow_torch = flow_torch.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            else:
                diff_filter = self.get_diff_filter(inchannels=1, precision="64")
                img_middle = img_middle.unsqueeze(1)
                # gradient_img = diff_filter(img_middle) / 2
                gradient_img = diff_filter(img_middle)

            diff_img = motion_fusion_optimized(
                gradient_img, flow_torch, delta_t, self.c
            )

            if self.cfg.is_color:
                diff_img = diff_img.permute(0, 2, 3, 1)

            if self.global_step % self.cfg.plot_interval == 0:
                if self.cfg.is_color:
                    flow = flow_torch[:, :, 0, :, :].detach().cpu().numpy()
                else:
                    flow = flow_torch.detach().cpu().numpy()
        else:
            raise NotImplementedError(f"Unknown diff_method: {self.cfg.diff_method}")

        return (
            diff_img,
            img_start,
            depth_middle_raw,
            depth_middle,
            gradient_img if "gradient_img" in locals() else None,
        )
