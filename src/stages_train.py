import os
import time
from pathlib import Path
import torch
from tqdm import tqdm

from src.config import (
    AccumulationMethod,
    BackgroundColor,
    EventSimMethod,
)
from src.utils.rotations import matrix_to_quaternion, quaternion_to_matrix
from src.costs import MultiFocalNormalizedGradientMagnitudeHuber
from src.third_party.threedgrut.threedgrut.utils.misc import check_step_condition
from src.utils.motion import compute_velocity_and_angular_velocity
from src.visualizer import plot_loss


def train_runner_stage(self):
    self.logger.info("Starting training...")
    start_time = time.time()

    # Use DataLoader with multiple workers for parallel data loading
    trainloader = torch.utils.data.DataLoader(
        self.dataset,
        shuffle=True,
        num_workers=self.num_workers,
    )
    trainloader_iter = iter(trainloader)

    pbar = tqdm(range(self.cfg.max_steps))

    # Initialize loss
    loss = torch.tensor(0.0, device=self.device)
    loss_history = {
        "loss": [],
        "focus_loss": [],
        "l1loss": [],
        "l2loss": [],
        "ssim_loss": [],
        "opacity_reg_loss": [],
        "scale_reg_loss": [],
        "variation_loss": [],
        "depth_variation_loss": [],
        "loss_map": {},
    }
    assert self.model.optimizer is not None, "Optimizer is not set in the model."
    self.model.optimizer.zero_grad()

    # Optimization loop
    for step in pbar:
        loss = torch.tensor(0.0, device=self.device)
        self.global_step = step
        self.renderer.global_step = self.global_step

        # Step-dependent processing
        if step == self.cfg.use_diff_image_step:
            self.use_diff_image = True
            self.event_processor.event_sim_method = EventSimMethod.POLARITY
            self.renderer.background_color = self.cfg.background_color
            # self.background_color = BackgroundColor.GRAY  # gray background for training

            if self.cfg.randomize_gaussian_color:
                # Randomize gaussian color
                from src.third_party.threedgrut.threedgrut.utils.misc import (
                    sh_degree_to_specular_dim,
                )

                num_gaussians = self.model.num_gaussians
                dtype = self.model.features_albedo.dtype
                fused_color = torch.rand(
                    (num_gaussians, 3), dtype=dtype, device=self.device
                )
                features_albedo = fused_color.contiguous()
                max_sh_degree = self.model.max_n_features
                num_specular_features = sh_degree_to_specular_dim(max_sh_degree)
                features_specular = torch.zeros(
                    (num_gaussians, num_specular_features),
                    dtype=dtype,
                    device=self.device,
                ).contiguous()

                self.model.features_albedo = torch.nn.Parameter(
                    features_albedo.to(dtype=dtype, device=self.device)
                )
                self.model.features_specular = torch.nn.Parameter(
                    features_specular.to(dtype=dtype, device=self.device)
                )

            # Reset optimizer
            self.model.setup_optimizer()

        # Try: index error handling
        try:
            batch = next(trainloader_iter)
        except StopIteration:
            self.logger.warning(StopIteration)
            trainloader_iter = iter(trainloader)
            continue
        except IndexError:
            self.logger.warning(IndexError)
            trainloader_iter = iter(trainloader)
            continue

        # Compute velocity and angular velocity in middle camera coordinates
        velocity, angular_velocity = compute_velocity_and_angular_velocity(
            batch["pose_start"], batch["pose_end"], batch["pose_middle"], batch["events"][:, :, 2], precision=self.precision
        )

        (
            event_frame,
            iwe,
            depth,
            depth_map_,
            iwe_mask,
            iwe_grays,
            event_frame_gray,
        ) = self.accumulate_events(
            batch["events"],
            batch["pose_start"],
            batch["pose_middle"],
            batch["pose_end"],
            self.accumulation_method,
            velocity,
            angular_velocity,
        )

        if self.accumulation_method == AccumulationMethod.EVENT_FRAME:
            accumulated_events = event_frame
        elif self.accumulation_method == AccumulationMethod.IWE:
            accumulated_events = iwe

        if not self.use_diff_image:
            # Avoid to increase too much gaussians in pre-optimization
            accumulated_events = accumulated_events.clamp(0, 1)

        if self.use_diff_image:
            diff_img, img_start, depth_map, depth_map_adjusted, img_gradients = (
                self.renderer.create_diff_image(
                    batch["pose_start"],
                    batch["pose_end"],
                    batch["pose_middle"],
                    batch["events"][:, :, 2],
                    self.diff_method,
                    velocity,
                    angular_velocity,
                )
            )

            if self.event_processor.event_sim_method == EventSimMethod.BILINEAR_VOTE:
                diff_img = torch.abs(diff_img)
        else:
            diff_img, depth_map, opacity_map = self.renderer.render_image(
                batch["pose_middle"], render_depth=True
            )
            depth_map_adjusted = depth_map / (opacity_map + 1e-6)
            assert isinstance(diff_img, torch.Tensor), "diff_img should be a tensor"

            diff_img = (
                diff_img[..., 0] * 0.2990
                + diff_img[..., 1] * 0.5870
                + diff_img[..., 2] * 0.1140
            )
            img_start = diff_img  # for variation loss

        if (
            self.use_diff_image is False
            and self.renderer.background_color == BackgroundColor.WHITE
        ):
            # Invert diff image for white background
            diff_img = 1.0 - diff_img

        # Apply mask if provided by dataset
        mask = batch.get("mask")
        if mask is not None:
            diff_img = diff_img * mask
            accumulated_events = accumulated_events * mask

        # Loss calculation (l1, l2, silhouette)
        if self.cfg.use_l1:
            if accumulated_events is None:
                continue
            l1loss = self.l1_cost.calculate(
                {
                    "diff_img": diff_img,
                    "accumulated_events": accumulated_events,
                    "use_diff_image": self.use_diff_image,
                    "bayered_diff": self.bayered_diff,
                    "is_color": self.cfg.is_color,
                    "normalize": self.cfg.normalize_l1,
                    "use_masked": self.cfg.use_masked_l1,
                    "mask_weight": self.cfg.l1_mask_weight,
                    "weight": self.cfg.l1_weight,
                    "iwe_mask": iwe_mask,
                }
            )
            loss = loss + l1loss
            loss_history["l1loss"].append(l1loss.item())

        if self.cfg.use_l2:
            if accumulated_events is None:
                continue
            l2loss = self.l2_cost.calculate(
                {
                    "diff_img": diff_img,
                    "accumulated_events": accumulated_events,
                    "use_diff_image": self.use_diff_image,
                    "bayered_diff": self.bayered_diff,
                    "is_color": self.cfg.is_color,
                    "normalize": self.cfg.normalize_l2,
                    "use_masked": self.cfg.use_masked_l2,
                    "mask_weight": self.cfg.l2_mask_weight,
                    "weight": self.cfg.l2_weight,
                    "iwe_mask": iwe_mask,
                }
            )

            loss = loss + l2loss
            loss_history["l2loss"].append(l2loss.item())

        if self.cfg.use_ssim:
            if accumulated_events is None:
                continue
            ssim_loss = self.ssim_cost.calculate(
                {
                    "diff_img": diff_img,
                    "accumulated_events": accumulated_events,
                    "use_diff_image": self.use_diff_image,
                    "bayered_diff": self.bayered_diff,
                    "is_color": self.cfg.is_color,
                    "weight": self.cfg.ssim_weight,
                    "iwe_mask": iwe_mask,
                }
            )

            loss = loss + ssim_loss
            loss_history["ssim_loss"].append(ssim_loss.item())

        if self.accumulation_method == AccumulationMethod.IWE and self.cfg.use_focus:
            iwe_gray = iwe_grays["middle"]
            event_frame_gray = event_frame_gray.detach()

            if isinstance(self.focus_cost, MultiFocalNormalizedGradientMagnitudeHuber):
                iwe_gray_forward = iwe_grays["last"]
                iwe_gray_backward = iwe_grays["first"]
                focus_loss = (
                    self.focus_cost.calculate(
                        {
                            "orig_iwe": event_frame_gray,
                            "middle_iwe": iwe_gray,
                            "forward_iwe": iwe_gray_forward,
                            "backward_iwe": iwe_gray_backward,
                            "omit_boundary": True,
                        }
                    )
                    * self.cfg.focus_weight
                )
            else:
                focus_loss = (
                    self.focus_cost.calculate(
                        {
                            "orig_iwe": event_frame_gray,
                            "iwe": iwe_gray.unsqueeze(0),
                            "omit_boundary": True,
                        }
                    )
                    * self.cfg.focus_weight
                )

            loss = loss + focus_loss
            loss_history["focus_loss"].append(focus_loss.item())
        else:
            loss_history["focus_loss"].append(0.0)

        # Regularizations
        if self.cfg.variation_weight > 0:
            # Variation loss to encourage smoothness in the accumulated events
            variation_loss = self.variation_cost.calculate(
                {"flow": img_start, "omit_boundary": True}
            )
            variation_loss = variation_loss * self.cfg.variation_weight
            loss = loss + variation_loss
            loss_history["variation_loss"].append(variation_loss.item())
            self.logger.info(f"Variation loss: {variation_loss.item():.4f}")

        if self.cfg.depth_variation_weight > 0 and "depth_map" in locals().keys():
            # Depth variation loss to encourage smoothness in the depth map
            depth_map = depth_map.to(dtype=self.precision)
            depth_variation_loss = self.variation_cost.calculate(
                {"flow": depth_map, "omit_boundary": True}
            )

            loss = loss + depth_variation_loss * self.cfg.depth_variation_weight
            loss_history["depth_variation_loss"].append(depth_variation_loss.item())
            self.logger.info(f"Depth variation loss: {depth_variation_loss.item():.4f}")

        if self.cfg.opacity_reg_weight > 0:
            opacity_reg_loss = (
                self.cfg.opacity_reg_weight
                * torch.abs(
                    torch.sigmoid(self.model.get_model_parameters()["density"])
                ).mean()
            )
            loss = loss + opacity_reg_loss
            loss_history["opacity_reg_loss"].append(opacity_reg_loss.item())

        if self.cfg.scale_reg_weight > 0:
            scale_reg_loss = (
                self.cfg.scale_reg_weight
                * torch.abs(
                    torch.exp(self.model.get_model_parameters()["scale"])
                ).mean()
            )
            loss = loss + scale_reg_loss
            loss_history["scale_reg_loss"].append(scale_reg_loss.item())

        # Plot loss
        loss_history["loss"].append(loss.item())
        desc = f"loss={loss.item():.3f}"
        pbar.set_description(desc)

        if step % self.cfg.plot_interval == 0:
            plot_loss(
                cost_history=loss_history,
                path=self.fig_path,
                ignored_keys=["t", "loss_per_view", "loss_map"],
            )

        for name, param in self.model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                self.logger.warning(f"NaN detected in gradients of {name}")

        if not torch.isfinite(loss).all():
            raise ValueError("Loss contain nan or inf.")

        loss.backward()

        # Post-backward strategy
        with torch.cuda.nvtx.range(f"train_{self.global_step}_post_bwd"):
            scene_updated = self.strategy.post_backward(
                step=self.global_step,
                scene_extent=self.cfg.scene_extent,
                train_dataset=self.dataset,
                batch=self.renderer.gpu_batch,
            )

        self.model.optimizer.step()
        self.model.optimizer.zero_grad()

        # Post-optimization strategy
        with torch.cuda.nvtx.range(f"train_{self.global_step}_post_opt"):
            self.strategy.post_optimizer_step(
                step=self.global_step,
                scene_extent=self.cfg.scene_extent,
                train_dataset=self.dataset,
                batch=self.renderer.gpu_batch,
            )

        # Update the SH if required
        if self.model.progressive_training and check_step_condition(
            self.global_step, 0, 1e6, self.model.feature_dim_increase_interval
        ):
            self.model.increase_num_active_features()

        # Update the BVH if required (for 3dgrt)
        if scene_updated or (
            self.cfg_3dgrut.model.bvh_update_frequency > 0
            and self.global_step % self.cfg_3dgrut.model.bvh_update_frequency == 0
        ):
            with torch.cuda.nvtx.range(f"train_{self.global_step}_bvh"):
                self.model.build_acc(rebuild=True)

        self.logger.info(f"num_gaussians: {self.model.num_gaussians}")
        if step % 100 == 0:  # Log every 100 steps
            self.logger.info(f"loss: {loss.item():.4f}")

        # Visualize imgs
        if self.cfg.debugging:
            imgs = {}
            if self.event_processor.event_sim_method == EventSimMethod.POLARITY:
                background = 128
            else:
                background = 0
            try:
                imgs["iwe"] = self.cfg.img_coeff * iwe + background
            except Exception as e:
                self.logger.warning(f"Cannot add 'iwe' to imgs: {e}")
            try:
                imgs["event_frame"] = self.cfg.img_coeff * event_frame + 128
                if self.cfg.is_color:
                    imgs["event_frame_r"] = (
                        self.cfg.img_coeff * event_frame[..., 0] + 128
                    )
                    imgs["event_frame_g"] = (
                        self.cfg.img_coeff * event_frame[..., 1] + 128
                    )
                    imgs["event_frame_b"] = (
                        self.cfg.img_coeff * event_frame[..., 2] + 128
                    )
            except Exception as e:
                self.logger.warning(f"Cannot add 'event_frame' to imgs: {e}")
            try:
                if self.event_processor.event_sim_method == EventSimMethod.POLARITY:
                    imgs["diff_img"] = self.cfg.img_coeff * diff_img + 128
                else:
                    imgs["diff_img"] = self.cfg.img_coeff * diff_img
            except Exception as e:
                self.logger.warning(f"Cannot add 'diff_img' to imgs: {e}")
            try:
                # RGB2BGR for cv2.imwrite
                if self.cfg.is_color and self.use_diff_image:
                    img_start = img_start[..., [2, 1, 0]]
                imgs["rendered_img"] = img_start * 255  # 255
            except Exception as e:
                self.logger.warning(f"Cannot add 'img_start' or 'img_end' to imgs: {e}")
            try:
                imgs["depth_map"] = depth_map.squeeze() * 100
            except Exception as e:
                self.logger.warning(f"Cannot add 'depth_map' to imgs: {e}")

            self.visualizer.save_imgs(
                imgs,
                interval=self.cfg.plot_interval,
                step=step,
                vmin=self.cfg.vmin,
                vmax=self.cfg.vmax,
            )

        if self.global_step % 10000 == 0:
            self.save_checkpoint()

    # Record training time
    total_time = time.time() - start_time
    self.logger.info(f"Training completed in {total_time / 60:.2f} minutes")

    # Export optimized gaussians
    if self.cfg.export_ingp:
        ingp_path = (
            self.cfg.ingp_path
            if self.cfg.ingp_path
            else os.path.join(self.output_dir, "export_last.ingp")
        )
        from src.third_party.threedgrut.threedgrut.export import INGPExporter

        exporter = INGPExporter()
        exporter.export(self.model, Path(ingp_path))
    if self.cfg.export_ply:
        ply_path = (
            self.cfg.ply_path
            if self.cfg.ply_path
            else os.path.join(self.output_dir, "export_last.ply")
        )
        from src.third_party.threedgrut.threedgrut.export import PLYExporter

        exporter = PLYExporter()
        exporter.export(self.model, Path(ply_path))
    self.save_checkpoint(last_checkpoint=True)
