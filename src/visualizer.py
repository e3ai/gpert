from typing import Optional, List
import os
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2
import torch


def to_uint8_img(img_tensor: torch.Tensor, scale: float) -> np.ndarray:
    img = img_tensor.detach().cpu().numpy()
    if img.ndim >= 3:
        img = img[0]
    img = 128 + img * scale
    return np.clip(img, 0, 255).astype(np.uint8)


def plot_loss(
    path: str, cost_history: dict[str, list], ignored_keys: list[str] | None = None
) -> None:
    ignored = set(ignored_keys or [])

    plt.figure()
    for key, values in cost_history.items():
        if key in ignored:
            continue
        plt.plot(np.array(values), label=key)
    plt.legend()
    plt.savefig(path)

    plt.figure()
    for key, values in cost_history.items():
        if key in ignored:
            continue
        plt.plot(np.array(values), label=key)
    plt.yscale("log")
    root, ext = os.path.splitext(path)
    log_path = f"{root}_log{ext}" if ext else f"{path}_log.png"
    plt.savefig(log_path)


class Visualizer:
    def __init__(self, output_dir: str) -> None:
        """Initialize the Visualizer with the output directory.

        Args:
            output_dir (str): Directory to save visualizations.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Default depth visualization settings
        self.depth_colormap = "viridis"
        self.depth_save_comparison = False
        self.depth_save_raw = False

    def save_depth_map_with_colorbar(
        self,
        depth_img: np.ndarray,
        step: Optional[int] = None,
        name: str = "depth_map",
        colormap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        save_img: bool = True,
        save_dir: Optional[str] = None,
    ) -> np.ndarray:
        """Create a colorized depth image (no labels/colorbar) matching input size.

        This returns an HxWx3 uint8 RGB image produced by applying the given
        colormap to the input depth. If ``save_img`` is True, it also saves the
        image to disk. No axes, titles, or colorbars are drawn.

        Args:
            depth_img: 2D depth map (H, W). If a larger array is given, it will be squeezed.
            step: Optional step index used in the filename when saving.
            name: Output filename prefix.
            colormap: Matplotlib colormap name (e.g., 'viridis', 'plasma', 'jet', 'turbo').
            vmin: Optional lower bound for normalization. If None, uses min of positive depths.
            vmax: Optional upper bound for normalization. If None, uses max depth.
            save_img: If True, save the resulting image to ``output_dir``.

        Returns:
            np.ndarray: Colorized depth image in RGB order, dtype=uint8, shape (H, W, 3).
        """
        # Ensure 2D
        if depth_img.ndim >= 3:
            depth_img = np.squeeze(depth_img)

        # Guard against empty/invalid depths when computing vmin/vmax
        valid = depth_img[np.isfinite(depth_img)]
        if valid.size == 0:
            # Fallback to zeros if everything is invalid
            depth_img = np.zeros_like(depth_img, dtype=np.float32)
            valid = np.array([0.0], dtype=np.float32)

        if vmin is None:
            pos = valid[valid > 0]
            vmin = float(pos.min()) if pos.size > 0 else float(valid.min())
        if vmax is None:
            vmax = float(valid.max())

        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            # Prevent divide-by-zero; create a tiny span
            vmax = float(vmin + 1e-6)

        # Normalize to [0,1]
        norm = (depth_img.astype(np.float64) - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0.0, 1.0)

        # Apply colormap directly to get exact HxW output (RGBA in [0,1])
        cmap = cm.get_cmap(colormap)
        rgba = cmap(norm)
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)  # HxWx3 (RGB)

        if save_img:
            # Save using OpenCV (expects BGR)
            target_dir = save_dir if save_dir is not None else self.output_dir
            os.makedirs(target_dir, exist_ok=True)
            img_path = os.path.join(
                target_dir,
                f"{name}_{step}.png" if step is not None else f"{name}.png",
            )
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, bgr)

        return rgb

    def save_imgs(
        self,
        imgs: dict[str, torch.Tensor],
        interval: int = 1,
        step: int = 0,
        vmin: float = 0,
        vmax: float = 10,
    ) -> None:
        """Save images to the output directory.

        Args:
            imgs (dict[str, torch.Tensor]): Dictionary of images to save.
            interval (int): Interval for saving images (>= 1).
            step (int): Current step.
            scale (int, optional): Scale factor for reshape pixel values to 0~255.
        """
        if step % interval != 0:
            return
        else:
            for name, img in imgs.items():
                category_dir = os.path.join(self.output_dir, name)
                os.makedirs(category_dir, exist_ok=True)

                if isinstance(img, torch.Tensor):
                    img = img.clone().to("cpu").detach().numpy()
                if img.ndim >= 3:
                    img = img[0].squeeze()  # remove batch dimension

                # Special handling for depth maps
                if "depth" in name.lower():
                    # Convert back from the *100 scaling applied in the main script
                    depth_img = img / 100.0
                    # Use enhanced depth visualization with viridis colormap
                    self.save_depth_map_with_colorbar(
                        depth_img,
                        step,
                        name=name,
                        colormap=self.depth_colormap,
                        vmin=vmin,
                        vmax=vmax,
                        save_dir=category_dir,
                    )
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                    img_path = os.path.join(category_dir, f"{name}_{step}.png")
                    cv2.imwrite(img_path, img)
