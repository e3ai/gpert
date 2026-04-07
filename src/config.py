import dataclasses
from typing import Optional
from enum import Enum


class LogMethod(Enum):
    LINLOG = "linlog"
    LOG = "log"


class DiffMethod(Enum):
    ONCE = "once"
    TWICE = "twice"


class DataloaderMethod(Enum):
    TIME = "time"
    NUM = "num"


class InterpMethod(Enum):
    LIN = "lin"
    SLERP = "slerp"


class EventSimMethod(Enum):
    BILINEAR_VOTE = "bilinear_vote"
    POLARITY = "polarity"


class AccumulationMethod(Enum):
    EVENT_FRAME = "event_frame"
    IWE = "iwe"


class DataType(Enum):
    ROBUST_E_NERF = "robust_e-nerf"
    ROBUST_E_NERF_TEST = "robust_e-nerf_test"


class BackgroundColor(Enum):
    BLACK = "black"
    WHITE = "white"
    GRAY = "gray"
    RANDOM = "random"


class BayerPattern(Enum):
    RGGB = "RGGB"
    BGGR = "BGGR"
    GRBG = "GRBG"
    GBRG = "GBRG"


@dataclasses.dataclass
class Config:
    # Path
    data_type: DataType = DataType.ROBUST_E_NERF
    data_root: str = ""
    event_name: str = "undistorted_events.npz"
    traj_name: str = "camera_poses.npz"
    camera_info_name: str = "camera_calibration.npz"
    outdir: str = ""
    grut_cfg_path: str = "./cfg/grut_config/configs"
    test_dir: str = ""
    test_pose: str = ""
    gsinit_method: str = "random"  # ["random", "checkpoint"]
    gsinit_ckpt_path: Optional[str] = None

    c: float = 1.0
    max_steps: int = 125
    accumulation_num: int = 5000
    accumulation_time: float = 0.1
    initial_gaussians: int = 10000
    use_diff_image_step: int = 1000
    diff_method: DiffMethod = DiffMethod.ONCE
    accumulation_method: AccumulationMethod = AccumulationMethod.EVENT_FRAME
    dataloader_method: DataloaderMethod = DataloaderMethod.NUM
    interp_method: InterpMethod = InterpMethod.LIN
    log_method: LogMethod = LogMethod.LINLOG  # Method for log intensity conversion
    log_eps: float = 0.00196078431  # 0.5/255
    linlog_threshold: float = 0.078  # v2e value
    background_color: BackgroundColor = (
        BackgroundColor.GRAY
    )  # [black, white, gray, random]
    xyz_max: float = 0.4
    xyz_min: float = -0.4
    plot_interval: int = 100

    train: bool = True
    test: bool = True

    # Geometric branch settings
    use_focus: bool = False
    focus_weight: float = 0.1
    depth_grad: str = "density"  # ["density", "posrot"]
    multi_iwe: bool = False  # Use multi-focus IWE for focus loss

    # Photometric branch settings
    ssim_weight: float = 10
    use_ssim: bool = True
    use_l1: bool = True
    use_l2: bool = True
    l1_weight: float = 1.0  # Weight for L1 loss
    l2_weight: float = 1.0  # Weight for L2 loss
    normalize_l1: bool = True  # Normalize L1 loss inputs
    normalize_l2: bool = False  # Normalize L2 loss inputs
    use_masked_l1: bool = False  # Use masked L1 loss
    use_masked_l2: bool = False  # Use masked L2 loss
    l1_mask_weight: float = 0.9  # Weight for masked L1 loss (similar to Event-3DGS)
    l2_mask_weight: float = 0.9  # Weight for masked L2 loss (similar to Event-3DGS)

    # Regularization settings
    depth_variation_weight: float = 0.0  # Weight for depth variation loss
    variation_weight: float = 0  # Weight for variation loss
    opacity_reg_weight: float = 0.0  # Weight for opacity regularization loss
    scale_reg_weight: float = 0.0  # Weight for scale regularization loss

    # Color settings
    is_color: bool = False
    bayer_pattern: BayerPattern = BayerPattern.RGGB  # One of {RGGB, BGGR, GRBG, GBRG}
    devide_g: bool = True  # Divide G1 and G2 by 2 in bayer pattern
    bayered_diff: bool = False  # Apply bayer pattern to diff image

    # Visualization settings
    img_coeff: float = 100
    vmin: float = 0.0  # Minimum value for depth visualization
    vmax: float = 10.0  # Maximum value for depth visualization

    strategy: Optional[dict] = None  # Strategy configuration from YAML

    # Others
    randomize_offset: bool = (
        True  # Randomize the starting offset when accumulating events
    )
    iwe_noise_std: float = 0.0  # from Event3DGS paper
    strategy_method: str = "GSStrategy"
    scene_extent: float = 1.0  # meters
    randomize_gaussian_color: bool = True
    num_workers: int = 0

    ingp_path: Optional[str] = None
    ply_path: Optional[str] = None
    gsinit_ingp_path: Optional[str] = None
    gsinit_colmap_path: Optional[str] = None
    filter_type: str = "diff_filter"  # [diff_filter, diff_filter_1px]

    export_ingp: bool = False
    export_ply: bool = True

    # debug
    debugging: bool = True

    # test
    test_step: int = 1
    test_h: Optional[int] = None
    test_w: Optional[int] = None

    @classmethod
    def from_dict(cls, config_dict):
        """
        Create a Config instance from a dictionary.
        """
        # Convert string to enum
        config_dict["diff_method"] = DiffMethod(config_dict["diff_method"])
        config_dict["dataloader_method"] = DataloaderMethod(
            config_dict["dataloader_method"]
        )
        config_dict["data_type"] = DataType(config_dict["data_type"])
        config_dict["interp_method"] = InterpMethod(config_dict["interp_method"])
        config_dict["accumulation_method"] = AccumulationMethod(
            config_dict["accumulation_method"]
        )
        config_dict["background_color"] = BackgroundColor(
            config_dict["background_color"]
        )
        config_dict["log_method"] = LogMethod(config_dict["log_method"])

        # Handle strategy configuration if present
        if "strategy" in config_dict:
            # Keep strategy as dict for later processing
            pass

        return cls(**config_dict)
