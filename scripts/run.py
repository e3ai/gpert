from pathlib import Path
import numpy as np
import torch
from typing import Any, Optional, Tuple
import os
import argparse
import dataclasses
import yaml
import logging
import sys
from hydra.experimental import (
    initialize_config_dir,
    compose,
)  # for hydra config loading
import random
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for _path in (PROJECT_ROOT, SRC_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)


from src.config import (
    Config,
    AccumulationMethod,
    EventSimMethod,
    BackgroundColor,
)  # noqa: E402
from src.event_processor import EventProcessor  # noqa: E402
from src.event_dataset import EventDataset  # noqa: E402
from src.visualizer import Visualizer  # noqa: E402
from src.metric import Metric  # noqa: E402

from src.costs import (
    EventL1Loss,
    EventL2Loss,
    EventSSIMLoss,
    NormalizedGradientMagnitudeHuber,
    MultiFocalNormalizedGradientMagnitudeHuber,
    TotalVariation,
)
from src.utils import rgb_to_bayer_mosaic
from src.setup_grut import initialize_gaussians, initialize_strategy
from src.renderer import Renderer
from src.stages_train import train_runner_stage
from src.stages_test import test_runner_stage


class Runner:
    def __init__(self, cfg: Config) -> None:
        self.eval_step = 0
        self.cfg = cfg
        self._set_global_seed(getattr(self.cfg, "seed", 0))
        self.precision = torch.float64
        self.num_workers = self.cfg.num_workers
        self.scale_offset = None  # for gamma correction
        self.accumulation_method = self.cfg.accumulation_method
        self.colmap_alignment: Optional[dict[str, Any]] = None

        self._setup_output_dirs()
        self._save_config()
        self._setup_logger()

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.logger.info(f"device: {self.device}")

        # Load data
        self.dataset = EventDataset(
            self.cfg, self.logger, self.device, 
        )

        # Setup gaussians
        cfg_dir_3dgrut = os.path.abspath(self.cfg.grut_cfg_path)
        cfg_path = "apps/nerf_synthetic_3dgrt.yaml"
        with initialize_config_dir(config_dir=cfg_dir_3dgrut):
            cfg_3dgrut = compose(
                config_name=cfg_path
            )  # load main config (e.g. config.yaml)

        self.model, self.colmap_alignment = initialize_gaussians(
            cfg_3dgrut,
            self.cfg,
            self.dataset,
            self.logger,
            self.device,
            self.output_dir,
        )
        self.strategy = initialize_strategy(
            cfg_3dgrut, self.cfg, self.model, self.logger
        )
        self.cfg_3dgrut = cfg_3dgrut
        self.global_step = 0
        self.diff_method = self.cfg.diff_method

        # Setup costs
        if self.cfg.multi_iwe:
            self.focus_cost = MultiFocalNormalizedGradientMagnitudeHuber(
                direction="minimize",
                store_history=True,
                cuda_available=True,
                precision="64",
            )
        else:
            self.focus_cost = NormalizedGradientMagnitudeHuber(
                direction="minimize",
                store_history=True,
                cuda_available=True,
                precision="64",
            )
        self.variation_cost = TotalVariation(
            direction="minimize", cuda_available=True, precision="64", in_channels=1
        )
        self.l1_cost = EventL1Loss(direction="minimize", store_history=True)
        self.l2_cost = EventL2Loss(
            direction="minimize", store_history=True, precision="64"
        )
        self.ssim_cost = EventSSIMLoss(direction="minimize", store_history=True)

        if self.cfg.use_diff_image_step != 0:
            self.use_diff_image = False
            initial_event_sim_method = (
                EventSimMethod.BILINEAR_VOTE
            )  # pre-optimization without polarity
        else:
            self.use_diff_image = True
            initial_event_sim_method = EventSimMethod.POLARITY
        initial_background_color = BackgroundColor.BLACK

        # Setup metrics (PSNR/SSIM/LPIPS)
        self.metric_eval = Metric().to(self.device).eval()

        self.event_processor = EventProcessor(
            cfg=self.cfg,
            device=self.device,
            intrinsics=self.dataset.intrinsics,
            dtype="64",
            event_sim_method=initial_event_sim_method,
        )
        self.renderer = Renderer(
            cfg=self.cfg,
            device=self.device,
            precision=self.precision,
            dataset=self.dataset,
            model=self.model,
            logger=self.logger,
            background_color=initial_background_color,
            intrinsics=self.dataset.intrinsics,
            c=torch.tensor(self.cfg.c),
            global_step=self.global_step,
        )

        self.visualizer = Visualizer(self.output_dir)

        if self.cfg.bayered_diff and self.cfg.is_color:
            self.bayered_diff = True
        else:
            self.bayered_diff = False

    def _setup_output_dirs(self) -> None:
        """Create output directories and store derived paths."""
        os.makedirs(self.cfg.outdir, exist_ok=True)
        seq_num = str(len(os.listdir(self.cfg.outdir)))
        self.output_dir = os.path.join(self.cfg.outdir, seq_num)
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_config(self) -> None:
        """Persist the resolved config to the output directory."""
        with open(os.path.join(self.output_dir, "config.yaml"), "w") as f:
            yaml.dump(dataclasses.asdict(self.cfg), f, default_flow_style=False)

    def _setup_logger(self) -> None:
        """Initialize logger and log file output."""
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, "log.txt"), mode="w"),
                logging.StreamHandler(sys.stdout),
            ],
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("fit_smpl_event")
        self.fig_path = os.path.join(self.output_dir, "figures.png")

    def _set_global_seed(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def accumulate_events(
        self,
        events: torch.Tensor,
        pose_start: torch.Tensor,
        pose_middle: torch.Tensor,
        pose_end: torch.Tensor,
        accumulation_method: AccumulationMethod,
        velocity: Optional[torch.Tensor] = None,
        angular_velocity: Optional[torch.Tensor] = None,
        save: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[dict[str, Any]],
        Optional[torch.Tensor],
    ]:
        """Accumulate events into event frame or IWE.
        Args:
            events (torch.Tensor): Events tensor of shape (B, N, [y,x,t,p]).
            pose_start (torch.Tensor): Pose at the start frame. (B, 7) where 7 = (x,y,z,w,x,y,z).
            pose_end (torch.Tensor): Pose at the end frame. (B, 7) where 7 = (x,y,z,w,x,y,z).
            accumulation_method (AccumulationMethod): Accumulation method.
        Returns:
            Tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
                Accumulated events, depth, depth_map, and optional IWE masks.
        """
        iwe = None
        depth = None
        depth_map = None
        iwe_mask = None
        iwe_grays = None
        event_frame_gray = None
        if accumulation_method == AccumulationMethod.EVENT_FRAME:
            event_frame = self.event_processor.create_eventframe(
                events, sigma=0, is_color=self.cfg.is_color
            )
            event_frame = torch.tensor(
                event_frame, device=self.device, dtype=self.precision
            )
            iwe_mask, _ = rgb_to_bayer_mosaic(
                event_frame,
                bayer_pattern=getattr(self.cfg, "bayer_pattern", "RGGB"),
            )

        elif accumulation_method == AccumulationMethod.IWE:
            if velocity is None or angular_velocity is None:
                raise ValueError(
                    "velocity and angular_velocity are required for IWE accumulation"
                )
            events_sec = events.clone().to(self.precision)
            events_sec[..., 2] = ((events[..., 2] - events[0, 0, 2]) / 1e9).to(
                self.precision
            )

            depth_raw, depth, depth_map = self.renderer.render_depth(
                events, pose_start, pose_end
            )

            (
                iwe,
                event_frame,
                flow_torch,
                depth_map_,
                iwe_mask,
                iwe_grays,
                event_frame_gray,
            ) = self.event_processor.create_iwe(
                events_sec,
                depth,
                velocity,
                angular_velocity,
                self.cfg.is_color,
                devide_g=self.cfg.devide_g,
                multi_iwe=self.cfg.multi_iwe,
            )
        else:
            raise NotImplementedError(
                f"Unknown accumulation method: {accumulation_method}"
            )

        # Add batch and channel dim
        if depth_map is not None:
            depth_map = depth_map.unsqueeze(0).unsqueeze(-1)

        return event_frame, iwe, depth, depth_map, iwe_mask, iwe_grays, event_frame_gray

    def save_checkpoint(self, last_checkpoint: bool = False):
        """Saves checkpoint to a path under {conf.out_dir}/{conf.experiment_name}.
        Args:
            last_checkpoint: If true, will update checkpoint title to 'last'.
                             Otherwise uses global step
        """
        global_step = self.global_step
        out_dir = self.output_dir
        parameters = self.model.get_model_parameters()
        parameters |= {"global_step": self.global_step}

        strategy_parameters = self.strategy.get_strategy_parameters()
        parameters = {**parameters, **strategy_parameters}

        ckpt_dir = os.path.join(out_dir, "ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(os.path.join(ckpt_dir, f"ours_{int(global_step)}"), exist_ok=True)
        if not last_checkpoint:
            ckpt_path = os.path.join(
                ckpt_dir, f"ours_{int(global_step)}", f"ckpt_{global_step}.pt"
            )
        else:
            ckpt_path = os.path.join(ckpt_dir, "ckpt_last.pt")
        torch.save(parameters, ckpt_path)
        self.logger.info(f'💾 Saved checkpoint to: "{os.path.abspath(ckpt_path)}"')

    def train(self):
        return train_runner_stage(self)

    @torch.no_grad()
    def test(self):
        return test_runner_stage(self)


def main(cfg: Config) -> None:
    runner = Runner(cfg)
    if cfg.train:
        runner.train()
    if cfg.test:
        runner.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./cfg/smpl_event.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
        cfg = Config.from_dict(config_dict)

    main(cfg)
