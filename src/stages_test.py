import os

import numpy as np
import torch
import json
import cv2

from src.config import DataType
from src.metric import eval_metrics
from src.utils.rotations import matrix_to_quaternion
from src.utils.motion import convert_coordinates
from src.utils.gamma_correction import gamma_correction
from src.types import Intrinsics


@torch.no_grad()
def test_runner_stage(self):
    """Render test sequence from test_dir and test_pose, compute metrics like PSNR, SSIM, LPIPS."""
    self.dataset.set_eval()
    self.renderer.background_color = self.cfg.background_color

    test_dir_path = os.path.join(self.cfg.data_root, "views", self.cfg.test_dir)
    test_pose_path = os.path.join(self.cfg.data_root, "views", self.cfg.test_pose)

    # If test pose is not found, skip testing
    if not os.path.exists(test_pose_path):
        self.logger.warning(
            f"Test pose file not found: {test_pose_path}. Skipping test."
        )
        return

    with open(test_pose_path, "r") as f:
        pose_data = json.load(f)

    intrinsics = pose_data.get("intrinsics", None)
    if intrinsics is not None:
        fx = intrinsics[0][0]
        fy = intrinsics[1][1]
        cx = intrinsics[0][2]
        cy = intrinsics[1][2]
        K = [fx, fy, cx, cy]
        self.logger.info(
            f"Resetting intrinsics for test set: fx={fx}, fy={fy}, cx={cx}, cy={cy}"
        )

        new_intrinsics = Intrinsics(
            fx=torch.tensor(fx, device=self.device, dtype=self.precision),
            fy=torch.tensor(fy, device=self.device, dtype=self.precision),
            cx=torch.tensor(cx, device=self.device, dtype=self.precision),
            cy=torch.tensor(cy, device=self.device, dtype=self.precision),
            image_width=self.cfg.test_w,
            image_height=self.cfg.test_h,
        )
        self.dataset.reset_intrinsics(new_intrinsics)
        self.renderer.reset_intrinsics(new_intrinsics)

    frames = pose_data.get("frames", [])
    if len(frames) == 0:
        self.logger.warning("No frames found in test pose file.")
        return

    metrics_acc = {"psnr": [], "ssim": [], "lpips": []}
    per_frame_rows = []
    pred_batches: list[torch.Tensor] = []
    gt_batches: list[torch.Tensor] = []
    frame_contexts: list[dict[str, object]] = []

    # Directory to save corrected renders
    rendered_test_seq_dir = os.path.join(self.output_dir, "rendered_test_seq")
    os.makedirs(rendered_test_seq_dir, exist_ok=True)

    frames = frames[:: self.cfg.test_step]

    for i, fr in enumerate(frames):
        # Resolve GT image path (NeRF-style file_path like ./test/r_0)
        file_path = fr.get("file_path", "")
        base_name = os.path.basename(file_path)
        if not base_name:
            base_name = f"r_{i}"
        preferred_exts = [".png", ".jpg", ".jpeg"]
        base_root, base_ext = os.path.splitext(base_name)
        candidate_names = []

        if base_ext:
            candidate_names.append(base_name)
            if base_ext.lower() not in preferred_exts:
                candidate_names.extend([base_root + ext for ext in preferred_exts])
        else:
            candidate_names.extend([base_name + ext for ext in preferred_exts])

        candidate_names.extend([f"{i}{ext}" for ext in preferred_exts])

        gt_img_path = None
        seen_candidates = set()
        for candidate in candidate_names:
            if candidate in seen_candidates:
                continue
            seen_candidates.add(candidate)
            candidate_path = os.path.join(test_dir_path, candidate)
            if os.path.exists(candidate_path):
                gt_img_path = candidate_path
                break

        if gt_img_path is None:
            fallback_display = (
                os.path.join(test_dir_path, candidate_names[0])
                if candidate_names
                else "unknown"
            )
            self.logger.warning(
                f"GT image not found for frame {i}: expected variants like {fallback_display}"
            )
            continue

        # Build pose tensor (B,7) from 4x4 transform_matrix (camera-to-world, OpenGL style)
        tm = fr.get("transform_matrix", None)
        if tm is None:
            self.logger.warning(f"transform_matrix missing for frame {i}; skipping")
            continue
        tm_np = np.array(tm, dtype=np.float64)
        if tm_np.shape != (4, 4):
            self.logger.warning(
                f"Invalid transform_matrix shape for frame {i}: {tm_np.shape}"
            )
            continue

        R_c2w_np = tm_np[:3, :3]
        t_w_np = tm_np[:3, 3]

        R_c2w = (
            torch.from_numpy(R_c2w_np)
            .to(self.device, dtype=self.precision)
            .unsqueeze(0)
        )  # (1,3,3)
        t_w = (
            torch.from_numpy(t_w_np).to(self.device, dtype=self.precision).unsqueeze(0)
        )  # (1,3)

        t_w = -t_w  # camera centric -> world centric
        R_c2w, t_w = convert_coordinates(
            R_c2w,
            t_w,
            data_type=DataType.ROBUST_E_NERF_TEST,
            device=self.device,
            precision=self.precision,
        )

        q_wxyz = matrix_to_quaternion(R_c2w)  # (1,4) w,x,y,z
        pose = torch.cat([t_w, q_wxyz], dim=1)  # (1,7)

        # Render
        rendered = self.renderer.render_image(pose, render_depth=False)
        assert isinstance(rendered, torch.Tensor)
        rendered = rendered[0]  # (H,W,1/3)

        # Prepare color tensors for metrics and brightness correction
        if self.cfg.is_color:
            if rendered.shape[-1] == 3:
                rendered_color = rendered
            elif rendered.shape[-1] == 1:
                rendered_color = rendered.squeeze(-1).unsqueeze(-1).expand(-1, -1, 3)
            else:
                raise ValueError()
        else:
            if rendered.shape[-1] == 3:
                rendered_color = (
                    rendered[..., 0] * 0.299
                    + rendered[..., 1] * 0.587
                    + rendered[..., 2] * 0.114
                )
                rendered_color = rendered_color.unsqueeze(-1)  # HWC
            elif rendered.shape[-1] == 1:
                rendered_color = rendered
            else:
                raise ValueError()

        H_r, W_r = rendered_color.shape[:2]

        # Load GT, fill transparent regions with white (for robust e-nerf visualization)
        gt_img_any = cv2.imread(gt_img_path, cv2.IMREAD_UNCHANGED)
        if gt_img_any is None:
            self.logger.warning(f"Failed to read GT image: {gt_img_path}")
            continue
        # Build RGB GT and valid mask
        valid_mask_np = None
        if gt_img_any.ndim == 3 and gt_img_any.shape[2] == 4:
            # if RGBA, alpha to white
            gt_rgb = gt_img_any[:, :, :3]
            gt_rgb = gt_rgb[:, :, [2, 1, 0]]  # BGR2RGB
            gt_alpha = gt_img_any[:, :, 3].astype(np.float32) / 255
            gt_rgb = (
                gt_rgb * gt_alpha[:, :, np.newaxis]
                + 255 * (1 - gt_alpha)[:, :, np.newaxis]
            )
        elif gt_img_any.ndim == 3 and gt_img_any.shape[2] == 3:
            gt_rgb = gt_img_any
            gt_rgb = gt_rgb[:, :, [2, 1, 0]]  # BGR2RGB
        elif gt_img_any.ndim == 2:
            gt_rgb = cv2.cvtColor(gt_img_any, cv2.COLOR_GRAY2RGB)
        else:
            self.logger.warning(f"Unsupported GT image shape: {gt_img_any.shape}")
            continue

        if not self.cfg.is_color:
            gt_rgb = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2GRAY)
            gt_rgb = np.expand_dims(gt_rgb, axis=-1)  # HWC

        # Resize GT and mask to rendered size
        if (gt_rgb.shape[0], gt_rgb.shape[1]) != (H_r, W_r):
            gt_rgb = cv2.resize(gt_rgb, (W_r, H_r), interpolation=cv2.INTER_AREA)
            if valid_mask_np is not None:
                valid_mask_np = (
                    cv2.resize(
                        valid_mask_np.astype(np.uint8),
                        (W_r, H_r),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    > 0
                )

        pred_color = rendered_color.to(device=self.device, dtype=torch.float32)

        gt_color = torch.from_numpy(gt_rgb).to(self.device, dtype=torch.float32)
        gt_color = gt_color / 256 + (0.5 / 256)  # dequantize

        pred_batches.append(pred_color.detach().to(torch.float32))
        gt_batches.append(gt_color.detach().to(torch.float32))
        frame_contexts.append(
            {
                "idx": i,
                "name": os.path.basename(gt_img_path),
                "base_name": base_name,
                "render_dir": rendered_test_seq_dir,
            }
        )

    if pred_batches:
        pred_stack = torch.stack(pred_batches, dim=0)  # BHWC
        gt_stack = torch.stack(gt_batches, dim=0)  # BHWC

        # Avoid OOM
        # while pred_stack.shape[0] > 200:
        #     pred_stack = pred_stack[::2]
        #     gt_stack = gt_stack[::2]

        corrected_stack, self.scale_offset = gamma_correction(pred_stack, gt_stack)

        for idx_frame, corrected in enumerate(corrected_stack):
            ctx = frame_contexts[idx_frame]
            gt_tensor = gt_stack[idx_frame]

            corrected = corrected.to(torch.float32)
            gt_tensor = gt_tensor.to(torch.float32)

            try:
                psnr_fu, ssim_fu, lpips_fu = eval_metrics(
                    self.metric_eval, corrected, gt_tensor, None, device=self.device
                )
                metrics_acc["psnr"].append(float(psnr_fu))
                metrics_acc["ssim"].append(float(ssim_fu))
                metrics_acc["lpips"].append(float(lpips_fu))
            except Exception as e:
                self.logger.warning(
                    f"Metric evaluation (RGB) failed at frame {ctx['idx']}: {e}"
                )
                psnr_fu, ssim_fu, lpips_fu = float("nan"), 0.0, float("nan")

            out_path = os.path.join(
                ctx["render_dir"], f"{ctx['base_name']}_rendered.png"
            )
            try:
                pred_np = corrected.detach().cpu().numpy()
                if pred_np.ndim == 3 and pred_np.shape[2] == 3:
                    pred_np_bgr = pred_np[..., [2, 1, 0]]
                else:
                    pred_np_bgr = pred_np
                pred_u8 = np.clip(pred_np_bgr * 255.0, 0.0, 255.0).astype(np.uint8)
                cv2.imwrite(out_path, pred_u8)
            except Exception as e:
                self.logger.warning(
                    f"Failed to save corrected render for frame {ctx['idx']} to {out_path}: {e}"
                )

            gt_out_path = os.path.join(ctx["render_dir"], f"{ctx['base_name']}_gt.png")
            try:
                gt_np = gt_tensor.detach().cpu().numpy()
                gt_np_bgr = (
                    gt_np[..., [2, 1, 0]]
                    if gt_np.ndim == 3 and gt_np.shape[2] == 3
                    else gt_np
                )
                cv2.imwrite(
                    gt_out_path,
                    np.clip(gt_np_bgr * 255.0, 0.0, 255.0).astype(np.uint8),
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to save GT image for frame {ctx['idx']} to {gt_out_path}: {e}"
                )

            per_frame_rows.append(
                {
                    "idx": ctx["idx"],
                    "name": ctx["name"],
                    "metrics": (float(psnr_fu), float(ssim_fu), float(lpips_fu)),
                }
            )

    # Write metrics summary
    metrics_path = os.path.join(self.output_dir, "test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Test Evaluation Results\n")
        f.write("=======================\n\n")

        def write_avg(title: str, values):
            if not values:
                return
            f.write(
                f"{title}: {float(np.nanmean(np.array(values, dtype=np.float64))):.6f}\n"
            )

        write_avg("Mean PSNR", metrics_acc["psnr"])
        write_avg("Mean SSIM", metrics_acc["ssim"])
        write_avg("Mean LPIPS", metrics_acc["lpips"])

        f.write(f"\nNumber of compared frames: {len(per_frame_rows)}\n\n")

        # Per-frame metrics
        f.write("Per-frame metrics (PSNR, SSIM, LPIPS)\n")
        for row in per_frame_rows:
            idx = row["idx"]
            name = row["name"]
            f.write(f"Frame {idx:3d} ({name})\n")
            if "metrics" in row:
                ps, ss, lp = row["metrics"]
                f.write(f"  Metrics: PSNR={ps:8.4f}, SSIM={ss:7.5f}, LPIPS={lp:7.5f}\n")
    self.logger.info(f"Saved test evaluation metrics to: {metrics_path}")
