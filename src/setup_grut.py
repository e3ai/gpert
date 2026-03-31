from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf

from .config import Config
from .third_party.threedgrut.threedgrut.model.model import MixtureOfGaussians
from .third_party.threedgrut.threedgrut.strategy.gs import GSStrategy


def initialize_gaussians(
    cfg: DictConfig,
    custom_cfg: Config,
    dataset,
    logger,
    device: torch.device,
    output_dir: str,
) -> Tuple[MixtureOfGaussians, Optional[dict[str, Any]]]:
    """Initialize 3D Gaussian parameters and return alignment metadata if available."""
    logger.info("Initializing 3D Gaussian parameters...")
    model = MixtureOfGaussians(cfg, scene_extent=1.0)
    colmap_alignment: Optional[dict[str, Any]] = None

    if custom_cfg.gsinit_method == "random":
        model.init_from_random_point_cloud(
            num_gaussians=custom_cfg.initial_gaussians,
            xyz_max=custom_cfg.xyz_max,
            xyz_min=custom_cfg.xyz_min,
        )
    elif custom_cfg.gsinit_method == "checkpoint":
        checkpoint = torch.load(
            custom_cfg.gsinit_ckpt_path, map_location=device, weights_only=False
        )
        model.init_from_checkpoint(checkpoint)
    else:
        raise NotImplementedError(
            f"Unknown initialization method: {custom_cfg.gsinit_method}"
        )

    model.build_acc()
    model.setup_optimizer()

    return model, colmap_alignment


def initialize_strategy(
    conf: DictConfig,
    custom_cfg: Config,
    model: MixtureOfGaussians,
    logger,
) -> GSStrategy:
    """Set pre-train / post-train iteration logic. i.e. densification / pruning"""
    if hasattr(custom_cfg, "strategy") and custom_cfg.strategy is not None:
        strategy_config = custom_cfg.strategy
        conf_copy = conf.copy()

        if isinstance(strategy_config, dict):
            strategy_config = OmegaConf.create(strategy_config)

        conf_copy.strategy = strategy_config
        strategy = GSStrategy(conf_copy, model)
        logger.info("🔆 Using GS strategy with custom config")
    else:
        strategy = GSStrategy(conf, model)
        logger.info("🔆 Using GS strategy")


    strategy.init_densification_buffer(checkpoint=None)
    return strategy
