import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import cast


class SobelTorch(nn.Module):
    """Sobel operator for pytorch, for divergence calculation.
        This is equivalent implementation of
        ```
        sobelx = cv2.Sobel(flow[0], cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(flow[1], cv2.CV_64F, 0, 1, ksize=3)
        dxy = (sobelx + sobely) / 8.0
        ```
    Args:
        ksize (int) ... Kernel size of the convolution operation.
        in_channels (int) ... In channles.
        cuda_available (bool) ... True if cuda is available.
    """

    def __init__(
        self,
        ksize: int = 3,
        in_channels: int = 2,
        cuda_available: bool = False,
        precision="32",
    ):
        super().__init__()
        self.cuda_available = cuda_available
        self.in_channels = in_channels
        self.filter_dx = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=ksize,
            stride=1,
            padding=0,
            bias=False,
        )
        self.filter_dy = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=ksize,
            stride=1,
            padding=0,
            bias=False,
        )
        if precision == "64":
            Gy = torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
            ).double()
            Gx = torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
            ).double()
        else:
            Gy = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
            Gx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])

        if self.cuda_available:
            Gx = Gx.cuda()
            Gy = Gy.cuda()

        self.filter_dx.weight = nn.Parameter(
            Gx.unsqueeze(0).unsqueeze(0), requires_grad=False
        )
        self.filter_dy.weight = nn.Parameter(
            Gy.unsqueeze(0).unsqueeze(0), requires_grad=False
        )

    def forward(self, img):
        """
        Args:
            img (torch.Tensor) ... [b x (2 or 1) x H x W]. The 2 ch is [h, w] direction.

        Returns:
            sobel (torch.Tensor) ... [b x (4 or 2) x (H - 2) x (W - 2)].
                4ch means Sobel_x on xdim, Sobel_y on ydim, Sobel_x on ydim, and Sobel_y on xdim.
                To make it divergence, run `(sobel[:, 0] + sobel[:, 1]) / 8.0`.
        """
        if self.in_channels == 2:
            ch0 = F.pad(img[..., [0], :, :], (1, 1, 1, 1), mode="replicate")
            ch1 = F.pad(img[..., [1], :, :], (1, 1, 1, 1), mode="replicate")
            dxx = self.filter_dx(ch0)
            dyy = self.filter_dy(ch1)
            dyx = self.filter_dx(ch1)
            dxy = self.filter_dy(ch0)
            return torch.cat([dxx, dyy, dyx, dxy], dim=1)
        elif self.in_channels == 1:
            single = F.pad(img[..., [0], :, :], (1, 1, 1, 1), mode="replicate")
            dx = self.filter_dx(single)
            dy = self.filter_dy(single)
            return torch.cat([dx, dy], dim=1)
        else:
            padded = [
                F.pad(img[..., [i], :, :], (1, 1, 1, 1), mode="replicate")
                for i in range(img.shape[1])
            ]
            dx = [self.filter_dx(p) for p in padded]
            dy = [self.filter_dy(p) for p in padded]
            return torch.cat(dx + dy, dim=1)


class OptimizedSobelTorch(nn.Module):
    """Optimized Sobel operator with caching and better performance."""

    def __init__(
        self,
        ksize: int = 3,
        in_channels: int = 2,
        cuda_available: bool = False,
        precision="32",
    ):
        super().__init__()
        self.cuda_available = cuda_available
        self.in_channels = in_channels

        # Create kernels once
        if precision == "64":
            dtype = torch.float64
            Gy = torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=dtype
            )
            Gx = torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=dtype
            )
        else:
            dtype = torch.float32
            Gy = torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=dtype
            )
            Gx = torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=dtype
            )

        # Store kernels as buffers (not parameters to avoid gradients)
        self.register_buffer("Gx", Gx.view(1, 1, 3, 3))
        self.register_buffer("Gy", Gy.view(1, 1, 3, 3))

    def _sobel_single_channel(self, img: torch.Tensor) -> torch.Tensor:
        """Apply Sobel filter to single channel image"""
        img_rep = F.pad(img, (1, 1, 1, 1), mode="replicate")
        dx = F.conv2d(img_rep, cast(torch.Tensor, self.Gx))
        dy = F.conv2d(img_rep, cast(torch.Tensor, self.Gy))
        return torch.cat([dx, dy], dim=1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): [B x C x H x W] input image

        Returns:
            torch.Tensor: [B x (2*C) x H x W] Sobel gradients
        """
        if self.in_channels == 1:
            # Single channel optimization
            return self._sobel_single_channel(img)
        elif self.in_channels == 2:
            # Two channel optimization with vectorized operations
            img_rep = F.pad(img, (1, 1, 1, 1), mode="replicate")
            gx = cast(torch.Tensor, self.Gx)
            gy = cast(torch.Tensor, self.Gy)
            dx = F.conv2d(img_rep, gx.expand(2, -1, -1, -1), groups=2)
            dy = F.conv2d(img_rep, gy.expand(2, -1, -1, -1), groups=2)
            return torch.cat([dx, dy], dim=1)
        else:
            # General case for arbitrary channels
            results = []
            for i in range(img.shape[1]):
                single_ch = img[:, i : i + 1, :, :]
                sobel_result = self._sobel_single_channel(single_ch)
                results.append(sobel_result)
            return torch.cat(results, dim=1)


class DiffTorch(nn.Module):
    """Optimized differential operator with caching and better performance."""

    def __init__(
        self,
        ksize: int = 3,
        in_channels: int = 2,
        cuda_available: bool = False,
        precision="32",
    ):
        super().__init__()
        self.cuda_available = cuda_available
        self.in_channels = in_channels

        # Create kernels once
        if precision == "64":
            dtype = torch.float64
            Gy = torch.tensor(
                [[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=dtype
            )
            Gx = torch.tensor(
                [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]], dtype=dtype
            )
        else:
            dtype = torch.float32
            Gy = torch.tensor(
                [[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=dtype
            )
            Gx = torch.tensor(
                [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]], dtype=dtype
            )

        # Store kernels as buffers (not parameters to avoid gradients)
        self.register_buffer("Gx", Gx.view(1, 1, 3, 3))
        self.register_buffer("Gy", Gy.view(1, 1, 3, 3))

    def _diff_single_channel(self, img: torch.Tensor) -> torch.Tensor:
        """Apply Sobel filter to single channel image"""
        img_rep = F.pad(img, (1, 1, 1, 1), mode="replicate")
        dx = F.conv2d(img_rep, cast(torch.Tensor, self.Gx))
        dy = F.conv2d(img_rep, cast(torch.Tensor, self.Gy))
        return torch.cat([dx, dy], dim=1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): [B x C x H x W] input image

        Returns:
            torch.Tensor: [B x (2*C) x H x W] xy gradients
        """
        if self.in_channels == 1:
            # Single channel optimization
            return self._diff_single_channel(img)
        elif self.in_channels == 2:
            # Two channel optimization with vectorized operations
            img_rep = F.pad(img, (1, 1, 1, 1), mode="replicate")
            gx = cast(torch.Tensor, self.Gx)
            gy = cast(torch.Tensor, self.Gy)
            dx = F.conv2d(img_rep, gx.expand(2, -1, -1, -1), groups=2)
            dy = F.conv2d(img_rep, gy.expand(2, -1, -1, -1), groups=2)
            return torch.cat([dx, dy], dim=1)
        else:
            # General case for arbitrary channels
            results = []
            for i in range(img.shape[1]):
                single_ch = img[:, i : i + 1, :, :]
                diff_result = self._diff_single_channel(single_ch)
                results.append(diff_result)
            return torch.cat(results, dim=1)


class DiffTorch1px(nn.Module):
    """1-pixel differential operator using forward differences.

    dx = I(x+1, y) - I(x, y)
    dy = I(x, y+1) - I(x, y)
    """

    def __init__(
        self,
        ksize: int = 3,
        in_channels: int = 2,
        cuda_available: bool = False,
        precision="32",
    ):
        super().__init__()
        self.cuda_available = cuda_available
        self.in_channels = in_channels

        if precision == "64":
            dtype = torch.float64
        else:
            dtype = torch.float32

        # 1px forward-difference kernels
        kx = torch.tensor([[-1.0, 1.0]], dtype=dtype)
        ky = torch.tensor([[-1.0], [1.0]], dtype=dtype)

        self.register_buffer("Kx", kx.view(1, 1, 1, 2))
        self.register_buffer("Ky", ky.view(1, 1, 2, 1))

    def _diff_single_channel(self, img: torch.Tensor) -> torch.Tensor:
        """Apply 1px forward differences to a single channel image."""
        # Right and bottom replicate padding preserve the original HxW.
        img_x = F.pad(img, (0, 1, 0, 0), mode="replicate")
        img_y = F.pad(img, (0, 0, 0, 1), mode="replicate")
        dx = F.conv2d(img_x, cast(torch.Tensor, self.Kx))
        dy = F.conv2d(img_y, cast(torch.Tensor, self.Ky))
        return torch.cat([dx, dy], dim=1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): [B x C x H x W] input image

        Returns:
            torch.Tensor: [B x (2*C) x H x W] 1px xy gradients
        """
        channels = img.shape[1]

        if channels == 1:
            return self._diff_single_channel(img)

        # Grouped convolution computes each channel independently.
        img_x = F.pad(img, (0, 1, 0, 0), mode="replicate")
        img_y = F.pad(img, (0, 0, 0, 1), mode="replicate")
        kx = cast(torch.Tensor, self.Kx)
        ky = cast(torch.Tensor, self.Ky)
        dx = F.conv2d(img_x, kx.expand(channels, -1, -1, -1), groups=channels)
        dy = F.conv2d(img_y, ky.expand(channels, -1, -1, -1), groups=channels)
        return torch.cat([dx, dy], dim=1)


_cached_diff_filters = {}


def get_cached_diff_filter(
    in_channels: int = 1,
    cuda_available: bool = True,
    device=None,
    precision: str = "32",
    filter_type: str = "diff_filter",
) -> nn.Module:
    """Get cached differential filter instance.

    Args:
        in_channels (int): Number of input channels.
        cuda_available (bool): Compatibility flag.
        device: Target device for the filter module.
        precision (str): "32" or "64".
        filter_type (str): "diff_filter" or "diff_filter_1px".
    """
    global _cached_diff_filters

    key = (
        filter_type,
        in_channels,
        cuda_available,
        precision,
        str(device) if device is not None else None,
    )

    if key not in _cached_diff_filters:
        if filter_type == "diff_filter":
            module = DiffTorch(
                in_channels=in_channels,
                cuda_available=cuda_available,
                precision=precision,
            )
        elif filter_type == "diff_filter_1px":
            module = DiffTorch1px(
                in_channels=in_channels,
                cuda_available=cuda_available,
                precision=precision,
            )
        else:
            raise ValueError(
                f"Unsupported filter_type: {filter_type}. "
                "Use 'diff_filter' or 'diff_filter_1px'."
            )

        if device is not None:
            module = module.to(device)

        _cached_diff_filters[key] = module

    return _cached_diff_filters[key]


def motion_fusion_optimized(
    gradient: torch.Tensor,
    flow_torch: torch.Tensor,
    delta_t: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    """Fused Sobel x flow computation (JIT compiled for speed)"""
    return -torch.sum(gradient * flow_torch * delta_t, dim=1) / c

__all__ = [
    "SobelTorch",
    "OptimizedSobelTorch",
    "DiffTorch",
    "DiffTorch1px",
    "get_cached_diff_filter",
    "motion_fusion_optimized",
]
