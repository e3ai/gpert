from typing import Optional, Tuple
import torch
import logging

from .config import Config, EventSimMethod, LogMethod
from .event_utils import EventImageConverter, linlog, warp_event_by_event
from .types import Intrinsics


class EventProcessor:
    """Handle event data processing and accumulation"""

    def __init__(
        self,
        cfg: Config,
        device: torch.device,
        intrinsics: Intrinsics,
        dtype: str = "32",
        event_sim_method: EventSimMethod = EventSimMethod.BILINEAR_VOTE,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.event_sim_method = event_sim_method
        self.logger = logging.getLogger("EventProcessor")
        self.H, self.W = intrinsics.image_height, intrinsics.image_width
        self.fx, self.fy = intrinsics.fx, intrinsics.fy
        if dtype == "32":
            self.precision = torch.float32
            self.fx = torch.tensor(self.fx, device=self.device, dtype=torch.float32)
            self.fy = torch.tensor(self.fy, device=self.device, dtype=torch.float32)
        elif dtype == "64":
            self.precision = torch.float64
            self.fx = torch.tensor(self.fx, device=self.device, dtype=torch.float64)
            self.fy = torch.tensor(self.fy, device=self.device, dtype=torch.float64)
        self.step = 0

        self.converter = EventImageConverter((self.H, self.W))
        self.event_sim_method = event_sim_method

    @staticmethod
    def _ensure_bhw(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x.unsqueeze(0)  # 1,H,W
        return x  # B,H,W

    def bayer_to_rgb(self, bayerd_image: torch.Tensor, pattern: str) -> torch.Tensor:
        """Convert single-channel event frame to 3-channel RGB based on Bayer pattern.

        Args:
            bayerd_image (torch.Tensor): Single-channel event frame of shape (H, W) or (B, H, W).
            pattern (str): One of {"RGGB", "BGGR", "GRBG", "GBRG"}.

        Returns:
            torch.Tensor: 3-channel RGB event frame of shape (H, W, 3) or (B, H, W, 3).
        """
        if pattern not in {"RGGB", "BGGR", "GRBG", "GBRG"}:
            raise ValueError(f"Unsupported Bayer pattern: {pattern}")

        had_batch = bayerd_image.ndim == 3
        if bayerd_image.ndim == 2:
            bayerd_image = bayerd_image.unsqueeze(0)

        # bayerd_image: (B,H,W). We will sample per Bayer to build RGB of same H,W in channel-last.
        bayerd_image = bayerd_image.to(self.device)

        B, H, W = bayerd_image.shape
        # Create empty RGB (B, H, W, 3)
        rgb = torch.zeros((B, H, W, 3), dtype=bayerd_image.dtype, device=self.device)

        # Index helpers (avoid out-of-bound when H/W are odd)
        h_even = slice(0, H, 2)
        h_odd = slice(1, H, 2)
        w_even = slice(0, W, 2)
        w_odd = slice(1, W, 2)

        if pattern == "RGGB":
            rgb[:, h_even, w_even, 0] = bayerd_image[:, h_even, w_even]  # R
            rgb[:, h_even, w_odd, 1] = bayerd_image[:, h_even, w_odd]  # G
            rgb[:, h_odd, w_even, 1] = bayerd_image[:, h_odd, w_even]  # G
            rgb[:, h_odd, w_odd, 2] = bayerd_image[:, h_odd, w_odd]  # B
        elif pattern == "BGGR":
            rgb[:, h_even, w_even, 2] = bayerd_image[:, h_even, w_even]  # B
            rgb[:, h_even, w_odd, 1] = bayerd_image[:, h_even, w_odd]  # G
            rgb[:, h_odd, w_even, 1] = bayerd_image[:, h_odd, w_even]  # G
            rgb[:, h_odd, w_odd, 0] = bayerd_image[:, h_odd, w_odd]  # R
        elif pattern == "GRBG":
            rgb[:, h_even, w_even, 1] = bayerd_image[:, h_even, w_even]  # G
            rgb[:, h_even, w_odd, 0] = bayerd_image[:, h_even, w_odd]  # R
            rgb[:, h_odd, w_even, 2] = bayerd_image[:, h_odd, w_even]  # B
            rgb[:, h_odd, w_odd, 1] = bayerd_image[:, h_odd, w_odd]  # G
        elif pattern == "GBRG":
            rgb[:, h_even, w_even, 1] = bayerd_image[:, h_even, w_even]  # G
            rgb[:, h_even, w_odd, 2] = bayerd_image[:, h_even, w_odd]  # B
            rgb[:, h_odd, w_even, 0] = bayerd_image[:, h_odd, w_even]  # R
            rgb[:, h_odd, w_odd, 1] = bayerd_image[:, h_odd, w_odd]  # G

        # If input had no batch, squeeze back to (H,W,3)
        if not had_batch:
            rgb = rgb.squeeze(0)
        return rgb

    def create_bayer_masks(
        self,
        pattern: str,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Bayer masks as (R, G1, G2, B) for event coords or full image grid."""

        if x is not None and y is not None:
            x_i, y_i = x.to(torch.int64), y.to(torch.int64)
            x_even, y_even = (x_i & 1) == 0, (y_i & 1) == 0
        elif height is not None and width is not None and device is not None:
            y_idx = torch.arange(height, device=device)
            x_idx = torch.arange(width, device=device)
            yy, xx = torch.meshgrid(y_idx, x_idx, indexing="ij")
            y_even = (yy % 2) == 0
            x_even = (xx % 2) == 0
        else:
            raise ValueError(
                "Provide either x and y, or height/width/device for Bayer masks"
            )

        y_odd = torch.logical_not(y_even)
        x_odd = torch.logical_not(x_even)
        pattern = pattern.upper()

        if pattern == "RGGB":
            mask_r = y_even & x_even
            mask_g1 = y_even & x_odd
            mask_g2 = y_odd & x_even
            mask_b = y_odd & x_odd
        elif pattern == "BGGR":
            mask_b = y_even & x_even
            mask_g1 = y_even & x_odd
            mask_g2 = y_odd & x_even
            mask_r = y_odd & x_odd
        elif pattern == "GRBG":
            mask_g1 = y_even & x_even
            mask_r = y_even & x_odd
            mask_b = y_odd & x_even
            mask_g2 = y_odd & x_odd
        elif pattern == "GBRG":
            mask_g1 = y_even & x_even
            mask_b = y_even & x_odd
            mask_r = y_odd & x_even
            mask_g2 = y_odd & x_odd
        else:
            raise ValueError(f"Unsupported Bayer pattern: {pattern}")

        return mask_r, mask_g1, mask_g2, mask_b

    def create_eventframe(
        self, events: torch.Tensor, sigma: int = 0, is_color: bool = False
    ) -> torch.Tensor:
        """
        Args:
            events (torch.Tensor): Events tensor of shape (B, N, [y,x,t,p])
            sigma (int, optional): Gaussian sigma for smoothing. Defaults to 0.
            is_color (bool, optional): Whether to convert to color using Bayer pattern. Defaults to False.
        Returns:
            torch.Tensor: Event frame tensor of shape (B, H, W) or (B, H, W, 3)"""
        method = self.event_sim_method
        event_frame = self.converter.create_eventframe(
            events, method=method.value, sigma=sigma
        )
        if not isinstance(event_frame, torch.Tensor):
            event_frame = torch.tensor(event_frame, device=self.device)

        # Add batch dim
        if event_frame.ndim == 3:
            event_frame = event_frame.unsqueeze(0)

        if method == EventSimMethod.POLARITY:
            if is_color:
                pattern = getattr(self.cfg, "bayer_pattern", "RGGB")
                pos = self.bayer_to_rgb(event_frame[:, 0], pattern)  # (B,H,W,3)
                neg = self.bayer_to_rgb(event_frame[:, 1], pattern)  # (B,H,W,3)
                event_frame = torch.stack([pos, neg], dim=1)  # (B,2,H,W,3)

            event_frame = event_frame[:, 0] - event_frame[:, 1]

        return event_frame

    def create_iwe(
        self,
        events_sec: torch.Tensor,
        depth: torch.Tensor,
        velocity: torch.Tensor,
        angular_velocity: torch.Tensor,
        # event_sim_method: EventSimMethod,
        is_color: bool = False,
        devide_g: bool = True,
        multi_iwe: bool = False,
    ) -> tuple:
        """Create Image of Warped Events (IWE) from events and poses.

        Args:
            events_sec (torch.Tensor): Events tensor of shape (B, N, [y,x,t,p]). Timestamp is in seconds.
            depth (torch.Tensor): Depth tensor of shape (B, N, 1) or (B, H, W, 1).
            pose_start (torch.Tensor): Pose at the start frame of shape (B, 7) where 7 = (x,y,z,w,x,y,z).
            pose_end (torch.Tensor): Pose at the end frame of shape (B, 7) where 7 = (x,y,z,w,x,y,z).

        Returns:
            IWE, event frame, flow tensor, depth map, active mask, inactive mask.
        """
        flow_torch = None
        warped_events = None
        depth_map = None

        # Generate depth map from depth + events
        h, w = self.H, self.W
        nb = events_sec.shape[0]  # number of batches
        depth_map = torch.zeros((nb, h, w), device=self.device)  # (B, H, W)

        # Add color channel to events (determine Bayer color per event BEFORE warp)
        # Note: events_sec is in (B, N, [y, x, t, p]) order in this codebase.
        pattern = getattr(self.cfg, "bayer_pattern", "RGGB") if is_color else None

        if is_color and self.event_sim_method is EventSimMethod.POLARITY:
            assert pattern is not None
            pattern_str = str(pattern)
            y_coords, x_coords = events_sec[..., 0], events_sec[..., 1]

            # Delete batch dim
            masks = self.create_bayer_masks(
                pattern=pattern_str, x=x_coords[0], y=y_coords[0]
            )  # (mask_r, mask_g1, mask_g2, mask_b)

        clipped_depth = torch.clip(depth.squeeze(-1), min=1e-6)

        warped_events, weight = warp_event_by_event(
            events=events_sec[..., [1, 0, 2, 3]],
            v=velocity,
            w=angular_velocity,
            z=clipped_depth,
            f=self.fx,
            image_size=(self.H, self.W),
            random_coeff=0.0,
            direction="middle",
            data_type=self.cfg.data_type.value,
            device=self.device,
        )

        iwe_gray_middle = self.converter.create_eventframe(
            warped_events, method="bilinear_vote", sigma=0
        )  # for focus

        iwe_grays = {"middle": iwe_gray_middle}
        event_frame_gray = self.converter.create_eventframe(
            events_sec[..., [1, 0, 2, 3]], method="bilinear_vote", sigma=0
        )  # for focus

        if multi_iwe:
            warped_events_first, _ = warp_event_by_event(
                events=events_sec[..., [1, 0, 2, 3]],
                v=velocity,
                w=angular_velocity,
                z=clipped_depth,
                f=self.fx,
                image_size=(self.H, self.W),
                random_coeff=0.0,
                direction="first",
                data_type=self.cfg.data_type.value,
                device=self.device,
            )

            warped_events_last, _ = warp_event_by_event(
                events=events_sec[..., [1, 0, 2, 3]],
                v=velocity,
                w=angular_velocity,
                z=clipped_depth,
                f=self.fx,
                image_size=(self.H, self.W),
                random_coeff=0.0,
                direction="last",
                data_type=self.cfg.data_type.value,
                device=self.device,
            )

            iwe_gray_first = self.converter.create_eventframe(
                warped_events_first, method="bilinear_vote", sigma=0
            )

            iwe_gray_last = self.converter.create_eventframe(
                warped_events_last, method="bilinear_vote", sigma=0
            )

            iwe_grays["first"] = iwe_gray_first
            iwe_grays["last"] = iwe_gray_last

        # Add batch dim if missing
        iwe_active_mask = torch.zeros(
            (1, self.H, self.W, 3), dtype=torch.bool, device=self.device
        )

        if is_color and self.event_sim_method is EventSimMethod.POLARITY:
            # Reconstruct IWE per color channel and combine
            # Current path assumes batch size == 1 for accumulation (consistent with surrounding code).
            iwe = torch.zeros((1, self.H, self.W, 3), device=self.device)

            # Reorder to (y,x,t,p) for the converter
            warped_yxtp = warped_events[..., [1, 0, 2, 3]]

            for i in range(4):
                warped_events_c = warped_yxtp[0, masks[i]]

                if warped_events_c.ndim == 1:
                    warped_events_c = warped_events_c.unsqueeze(0)

                iwe_c = self.converter.create_eventframe(
                    warped_events_c.unsqueeze(0),
                    method=self.event_sim_method.value,
                    sigma=0,
                )

                iwe_active_mask_c = (iwe_c[0] != 0) | (iwe_c[1] != 0)
                iwe_c = iwe_c[0] - iwe_c[1]

                # Normalize to (H, W)
                if not isinstance(iwe_c, torch.Tensor):
                    iwe_c = torch.tensor(
                        iwe_c, device=self.device, dtype=self.precision
                    )
                if iwe_c.ndim == 3 and iwe_c.shape[0] == 1:
                    iwe_c = iwe_c.squeeze(0)

                if i == 0:  # R
                    iwe[0, :, :, 0] = iwe_c
                    iwe_active_mask[0, :, :, 0] = iwe_active_mask_c
                elif i == 1 or i == 2:  # G1 or G2:
                    if devide_g:
                        iwe[0, :, :, 1] += iwe_c / 2.0
                    else:
                        iwe[0, :, :, 1] += iwe_c
                    iwe_active_mask[0, :, :, 1] |= iwe_active_mask_c
                elif i == 3:  # B
                    iwe[0, :, :, 2] = iwe_c
                    iwe_active_mask[0, :, :, 2] = iwe_active_mask_c

        elif not is_color and self.event_sim_method is EventSimMethod.POLARITY:
            # Polarity IWE without color
            iwe = self.create_eventframe(
                warped_events[..., [1, 0, 2, 3]], sigma=0, is_color=False
            )
        else:
            iwe = self.create_eventframe(
                warped_events[..., [1, 0, 2, 3]], sigma=0, is_color=is_color
            )

        event_frame = self.create_eventframe(events_sec, sigma=0, is_color=is_color)

        if not isinstance(event_frame, torch.Tensor):
            event_frame = torch.tensor(
                event_frame, device=self.device, dtype=self.precision
            )

        if is_color and self.event_sim_method is EventSimMethod.POLARITY:
            iwe_zero = iwe == 0
            iwe_inactive_mask = torch.zeros_like(iwe_zero, dtype=torch.bool)
            iwe_inactive_mask[..., 0] = iwe_zero[..., 0]
            iwe_inactive_mask[..., 1] = iwe_zero[..., 1]
            iwe_inactive_mask[..., 2] = iwe_zero[..., 2]

            iwe_mask = torch.logical_or(iwe_active_mask, iwe_inactive_mask)
        else:
            iwe_mask = torch.ones_like(iwe, dtype=torch.bool, device=self.device)

        # Convert to torch.Tensor if needed
        if not isinstance(iwe, torch.Tensor):
            iwe = torch.tensor(iwe, device=self.device, dtype=self.precision)

        return (
            iwe,
            event_frame,
            flow_torch,
            depth_map,
            iwe_mask,
            iwe_grays,
            event_frame_gray,
        )
