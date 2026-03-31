"""
This code is based on robust-e-nerf (https://github.com/wengflow/robust-e-nerf/blob/main/scripts/tum_vie_to_esim.py).
It also supports event undistortion and chunked processing.

Usage example:
python scripts/tum_vie_to_esim.py mocap-desk2 /path/to/raw /path/to/preprocessed \
  --undistorted_img_width 1024 \
  --undistorted_img_height 1024
"""


import os
import sys
import argparse
import json

import easydict
import h5py
import hdf5plugin
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import cv2

# insert the project / script parent directory into the module search path
PROJECT_DIR = os.path.join(sys.path[0], '..')
sys.path.insert(1, PROJECT_DIR)
from src.types.trajectory import Trajectory
from src.rotations import quaternion_to_matrix
from src.config import InterpMethod


T_CCOMMON_COPENGL = np.array([[1,  0,  0, 0],
                              [0, -1,  0, 0],
                              [0,  0, -1, 0],
                              [0,  0,  0, 1]], dtype=np.float32)
US_TO_NS = int(1e+3)

# calibration constants
CAMERA_CALIBRATION_CONFIG_FILENAME_FORMAT_STR = "camera-calibration{}.json"
MOCAP_IMU_CALIBRATION_CONFIG_FILENAME_FORMAT_STR = (
    "mocap-imu-calibration{}.json"
)
SEQUENCE_NAMES_WITH_CONFIG_ID_A = [
    "loop-floor0", "loop-floor1", "loop-floor2", "loop-floor3",
    "mocap-desk", "mocap-desk2", "skate-easy"
]

CAMERA_POSITIONS = [ "left", "right" ]
CAMERA_INDICES = easydict.EasyDict({
    "rgb": { "left": 0,
             "right": 1 },
    "event": { "left": 2,
               "right": 3 }
})
TRIM_INITIAL_NUM_IMAGES = 80

# raw dataset file/folder names
RAW_EVENTS_FILENAME_FORMAT_STR = "{}-events_{}.h5"
NON_RAW_EVENTS_FOLDER_NAME_FORMAT_STR = "{}-vi_gt_data"
MARKER_POSES_FILENAME = "mocap_data.txt"
DISTORTED_IMAGES_FOLDER_NAME_FORMAT_STR = "{}_images"
IMAGE_TIMESTAMPS_FILENAME_FORMAT_STR = "image_timestamps_{}.txt"
IMAGE_FILENAME_FORMAT_STR = "{:05d}.jpg"

# preprocessed dataset file/folder names
PREPROCESSED_EVENTS_FILENAME = "raw_events.npz"
UNDISTORTED_EVENTS_FILENAME = "undistorted_events.npz"
PREPROCESSED_EVENT_CAMERA_POSES_FILENAME = "camera_poses.npz"
PREPROCESSED_EVENT_CAMERA_CALIBRATION_FILENAME = "camera_calibration.npz"
POSED_UNDISTORTED_IMAGES_FOLDER_NAME = "views"
STAGE = "val"
STAGE_TRANSFORMS_FILENAME_FORMAT_STR = "transforms_{}.json"

# assumed / estimated event camera parameters
ESTIMATED_REFRACTORY_PERIOD = 1375
ASSUMED_NEGATIVE_CONTRAST_THRESHOLD = 0.25
ESTIMATED_POSITIVE_TO_NEGATIVE_CONTRAST_THRESHOLD_RATIO = 1.458
NULL_BAYER_PATTERN = ""     # ie. monochrome camera
MASK_MORPH_KERNEL_SIZE = 3
MASK_MORPH_ITERATIONS = 1


def main(args):
    # derive config ID, non-raw events path & camera indices from user input
    if args.sequence_name in SEQUENCE_NAMES_WITH_CONFIG_ID_A:
        config_id = "A"
    else:
        config_id = "B"
    non_raw_events_path = os.path.join(
        args.raw_dataset_path,
        NON_RAW_EVENTS_FOLDER_NAME_FORMAT_STR.format(args.sequence_name)
    )
    # non_raw_events_path =args.raw_dataset_path
    rgb_cam_idx = CAMERA_INDICES.rgb[args.camera_position]
    event_cam_idx = CAMERA_INDICES.event[args.camera_position]

    # create the preprocessed dataset directory, if necessary
    os.makedirs(args.preprocessed_dataset_path, exist_ok=True)

    # load the TUM-VIE calibration results from the config file
    camera_calibration_path = os.path.join(
        args.raw_dataset_path,
        CAMERA_CALIBRATION_CONFIG_FILENAME_FORMAT_STR.format(config_id)
    )
    with open(camera_calibration_path) as f:
        camera_calibration = easydict.EasyDict(json.load(f))
    camera_calibration = camera_calibration.value0

    mocap_imu_calibration_path = os.path.join(
        args.raw_dataset_path,
        MOCAP_IMU_CALIBRATION_CONFIG_FILENAME_FORMAT_STR.format(config_id)
    )
    with open(mocap_imu_calibration_path) as f:
        mocap_imu_calibration = easydict.EasyDict(json.load(f))
    mocap_imu_calibration = mocap_imu_calibration.value0

    # derive event camera calibration parameters from raw TUM-VIE calibration
    ev_intr_dist = camera_calibration.intrinsics[event_cam_idx].intrinsics

    event_intrinsics = np.array(                                                # (3, 3)
        [[ ev_intr_dist.fx, 0,               ev_intr_dist.cx ],
         [ 0,               ev_intr_dist.fy, ev_intr_dist.cy ],
         [ 0,               0,               1               ]],
        dtype=np.float32
    )
    event_distortion_params = np.array(                                         # (4)
        [ ev_intr_dist.k1, ev_intr_dist.k2, ev_intr_dist.k3, ev_intr_dist.k4 ],
        dtype=np.float32
    )
    event_distortion_model = np.array({
        "kb4": "equidistant"
    }[camera_calibration.intrinsics[event_cam_idx].camera_type])

    event_img_width, event_img_height = (
        camera_calibration.resolution[event_cam_idx]
    )
    event_img_width = int(event_img_width)
    event_img_height = int(event_img_height)

    # derive RGB camera intrinsics first and use the undistorted RGB intrinsics as
    # the target event intrinsics.
    assert camera_calibration.intrinsics[rgb_cam_idx].camera_type == "kb4"
    rgb_intr_dist = camera_calibration.intrinsics[rgb_cam_idx].intrinsics

    rgb_intrinsics = np.array(                                                  # (3, 3)
        [[ rgb_intr_dist.fx, 0,                rgb_intr_dist.cx ],
         [ 0,                rgb_intr_dist.fy, rgb_intr_dist.cy ],
         [ 0,                0,                1                ]],
        dtype=np.float32
    )
    rgb_distortion_params = np.array(                                           # (4)
        [ rgb_intr_dist.k1, rgb_intr_dist.k2,
          rgb_intr_dist.k3, rgb_intr_dist.k4 ],
        dtype=np.float32
    )
    rgb_img_width, rgb_img_height = camera_calibration.resolution[rgb_cam_idx]
    new_rgb_intrinsics = (                                                      # (3, 3)
        cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            rgb_intrinsics, rgb_distortion_params,
            (rgb_img_width, rgb_img_height), R=np.eye(3, dtype=np.float32),
            balance=0
        )
    )

    target_intrinsics = new_rgb_intrinsics.astype(np.float32)

    target_img_width = (
        args.undistorted_img_width
        if args.undistorted_img_width is not None
        else int(rgb_img_width)
    )
    target_img_height = (
        args.undistorted_img_height
        if args.undistorted_img_height is not None
        else int(rgb_img_height)
    )

    neg_contrast_threshold = np.array(
        ASSUMED_NEGATIVE_CONTRAST_THRESHOLD, dtype=np.float32
    )
    pos_contrast_threshold = (
        ESTIMATED_POSITIVE_TO_NEGATIVE_CONTRAST_THRESHOLD_RATIO
        * neg_contrast_threshold
    )
    refractory_period = np.array(ESTIMATED_REFRACTORY_PERIOD, dtype=np.float32)
    bayer_pattern = NULL_BAYER_PATTERN

    # convert & save event camera poses into an npz file
    marker_poses_path = os.path.join(
        non_raw_events_path, MARKER_POSES_FILENAME
    )
    preprocessed_event_poses_path = os.path.join(
        args.preprocessed_dataset_path,
        PREPROCESSED_EVENT_CAMERA_POSES_FILENAME
    )
    marker_poses = np.loadtxt(marker_poses_path)                                # (P, 8)

    T_wm_timestamp = US_TO_NS * marker_poses[:, 0]                              # (P)
    T_wm_timestamp = T_wm_timestamp.astype(np.int64)
    T_wm = marker_poses[:, 1:].astype(np.float32)                               # (P, 7)
    T_wm = se3_vec_to_mat(T_wm)                                                 # (P, 4, 4)

    # trim the sequence to the interval-of-interest
    is_valid_timestamp = (args.start_timestamp <= T_wm_timestamp) \
                         & (T_wm_timestamp < args.end_timestamp)                # (P)
    T_wm_timestamp = T_wm_timestamp[is_valid_timestamp]                         # (P')
    init_T_wm_timestamp = T_wm_timestamp[0]
    T_wm_timestamp = T_wm_timestamp - init_T_wm_timestamp
    T_wm = T_wm[is_valid_timestamp, :, :]                                       # (P', 4, 4)

    # transform marker poses & timestamps to event camera poses & timestamps
    init_T_wc_timestamp = init_T_wm_timestamp
    T_wc_timestamp = T_wm_timestamp

    T_imu_marker = mocap_imu_calibration.T_imu_marker
    T_imu_marker = se3_json_to_mat(T_imu_marker)                                # (4, 4)

    T_imu_event = camera_calibration.T_imu_cam[event_cam_idx]
    T_imu_event = se3_json_to_mat(T_imu_event)                                  # (4, 4)

    T_marker_event = np.linalg.inv(T_imu_marker) @ T_imu_event                  # (4, 4)
    T_wc = T_wm @ T_marker_event                                                # (P', 4, 4)
    T_wc = se3_mat_to_vec(T_wc)                                                 # (P', 7)
    T_wc_position = T_wc[:, :3]                                                 # (P', 3)
    T_wc_orientation = T_wc[:, 3:]                                              # (P', 4)

    np.savez(
        preprocessed_event_poses_path,
        T_wc_position=T_wc_position,
        T_wc_orientation=T_wc_orientation,
        T_wc_timestamp=T_wc_timestamp
    )

    # convert & save events into an npz file
    raw_events_path = os.path.join(
        args.raw_dataset_path,
        RAW_EVENTS_FILENAME_FORMAT_STR.format(
            args.sequence_name, args.camera_position
        )
    )
    preprocessed_events_path = os.path.join(
        args.preprocessed_dataset_path, PREPROCESSED_EVENTS_FILENAME
    )
    undistorted_events_path = os.path.join(
        args.preprocessed_dataset_path, UNDISTORTED_EVENTS_FILENAME
    )
    event_mask_path = os.path.join(
        args.preprocessed_dataset_path,
        "mask.png",
    )

    if args.event_chunk_size <= 0:
        raise ValueError("event_chunk_size must be positive.")

    timestamp_lower = T_wc_timestamp[0]
    timestamp_upper = T_wc_timestamp[-1]
    undistort_intrinsics = event_intrinsics.astype(np.float64)
    undistort_distortion = event_distortion_params.astype(np.float64)
    undistort_target_intrinsics = target_intrinsics.astype(np.float64)
    temp_raw_position_path = os.path.join(
        args.preprocessed_dataset_path, "_tmp_raw_event_position.npy"
    )
    temp_undistorted_position_path = os.path.join(
        args.preprocessed_dataset_path, "_tmp_undistorted_event_position.npy"
    )
    temp_timestamp_path = os.path.join(
        args.preprocessed_dataset_path, "_tmp_event_timestamp.npy"
    )
    temp_polarity_path = os.path.join(
        args.preprocessed_dataset_path, "_tmp_event_polarity.npy"
    )
    temp_paths = (
        temp_raw_position_path,
        temp_undistorted_position_path,
        temp_timestamp_path,
        temp_polarity_path,
    )
    # Stream events in chunks to avoid loading the full HDF5 dataset into memory and undistort on the fly.
    raw_position_memmap = None
    undistorted_position_memmap = None
    timestamp_memmap = None
    polarity_memmap = None
    event_hit_mask = np.zeros(
        (target_img_height, target_img_width), dtype=np.uint8
    )
    try:
        with h5py.File(raw_events_path, "r") as f:
            events_group = f["events"]
            num_events = events_group["x"].shape[0]
            raw_position_dtype = events_group["x"].dtype

            total_valid_events = 0
            for start in range(0, num_events, args.event_chunk_size):
                end = min(start + args.event_chunk_size, num_events)
                t_chunk = np.asarray(events_group["t"][start:end])
                if np.issubdtype(t_chunk.dtype, np.integer):
                    timestamps_ns = US_TO_NS * t_chunk.astype(np.int64, copy=False)
                else:
                    timestamps_ns = np.rint(
                        t_chunk.astype(np.float64, copy=False) * US_TO_NS
                    ).astype(np.int64)
                timestamps_ns = timestamps_ns - init_T_wc_timestamp
                valid_mask = (
                    (timestamp_lower <= timestamps_ns)
                    & (timestamps_ns <= timestamp_upper)
                )
                total_valid_events += int(valid_mask.sum())

            raw_position_memmap = np.lib.format.open_memmap(
                temp_raw_position_path,
                mode="w+",
                dtype=raw_position_dtype,
                shape=(total_valid_events, 2)
            )
            undistorted_position_memmap = np.lib.format.open_memmap(
                temp_undistorted_position_path,
                mode="w+",
                dtype=np.float32,
                shape=(total_valid_events, 2)
            )
            timestamp_memmap = np.lib.format.open_memmap(
                temp_timestamp_path,
                mode="w+",
                dtype=np.int64,
                shape=(total_valid_events,)
            )
            polarity_memmap = np.lib.format.open_memmap(
                temp_polarity_path,
                mode="w+",
                dtype=np.bool_,
                shape=(total_valid_events,)
            )

            offset = 0
            for start in range(0, num_events, args.event_chunk_size):
                end = min(start + args.event_chunk_size, num_events)
                x_chunk = np.asarray(events_group["x"][start:end])
                y_chunk = np.asarray(events_group["y"][start:end])
                t_chunk = np.asarray(events_group["t"][start:end])
                p_chunk = np.asarray(events_group["p"][start:end], dtype=np.bool_)

                if np.issubdtype(t_chunk.dtype, np.integer):
                    timestamps_ns = US_TO_NS * t_chunk.astype(np.int64, copy=False)
                else:
                    timestamps_ns = np.rint(
                        t_chunk.astype(np.float64, copy=False) * US_TO_NS
                    ).astype(np.int64)
                timestamps_ns = timestamps_ns - init_T_wc_timestamp

                valid_mask = (
                    (timestamp_lower <= timestamps_ns)
                    & (timestamps_ns <= timestamp_upper)
                )
                time_valid_count = int(valid_mask.sum())
                if time_valid_count == 0:
                    continue

                timestamps_valid = timestamps_ns[valid_mask]
                x_valid = x_chunk[valid_mask]
                y_valid = y_chunk[valid_mask]
                p_valid = p_chunk[valid_mask]

                undistorted_valid = undistort_event_positions(
                    x_valid,
                    y_valid,
                    undistort_intrinsics,
                    undistort_distortion,
                    undistort_target_intrinsics,
                )
                in_bounds = compute_in_bounds_mask(
                    undistorted_valid,
                    target_img_width,
                    target_img_height,
                )
                valid_count = int(in_bounds.sum())
                if valid_count == 0:
                    continue

                timestamps_valid = timestamps_valid[in_bounds]
                x_valid = x_valid[in_bounds]
                y_valid = y_valid[in_bounds]
                p_valid = p_valid[in_bounds]
                undistorted_valid = undistorted_valid[in_bounds]

                # Build mask only from actually observed events that survived filtering.
                undistorted_x_pix = np.floor(undistorted_valid[:, 0]).astype(np.int64)
                undistorted_y_pix = np.floor(undistorted_valid[:, 1]).astype(np.int64)
                event_hit_mask[undistorted_y_pix, undistorted_x_pix] = 255

                raw_position_memmap[offset:offset + valid_count, 0] = x_valid
                raw_position_memmap[offset:offset + valid_count, 1] = y_valid
                timestamp_memmap[offset:offset + valid_count] = timestamps_valid
                polarity_memmap[offset:offset + valid_count] = p_valid
                undistorted_position_memmap[offset:offset + valid_count] = undistorted_valid
                offset += valid_count

        raw_position_memmap.flush()
        undistorted_position_memmap.flush()
        timestamp_memmap.flush()
        polarity_memmap.flush()

        np.savez(
            preprocessed_events_path,
            position=raw_position_memmap[:offset],
            timestamp=timestamp_memmap[:offset],
            polarity=polarity_memmap[:offset],
        )
        np.savez(
            undistorted_events_path,
            position=undistorted_position_memmap[:offset],
            timestamp=timestamp_memmap[:offset],
            polarity=polarity_memmap[:offset],
        )
        event_hit_mask = remove_small_black_dots(event_hit_mask)
        if not cv2.imwrite(event_mask_path, event_hit_mask):
            raise RuntimeError(f"Failed to write event mask image: {event_mask_path}")
    finally:
        for memmap in (
            raw_position_memmap,
            undistorted_position_memmap,
            timestamp_memmap,
            polarity_memmap,
        ):
            if memmap is not None and hasattr(memmap, "_mmap"):
                memmap._mmap.close()
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # derive & save RGB camera intrinsics & poses into a json file

    preprocessed_event_calibration_path = os.path.join(
        args.preprocessed_dataset_path,
        PREPROCESSED_EVENT_CAMERA_CALIBRATION_FILENAME
    )
    # Events are rectified with target intrinsics, so distortion becomes identity.
    np.savez(
        preprocessed_event_calibration_path,
        intrinsics=target_intrinsics.astype(np.float32),
        distortion_params=np.zeros_like(event_distortion_params),
        distortion_model=np.array("none"),
        img_height=np.array(target_img_height, dtype=np.uint16),
        img_width=np.array(target_img_width, dtype=np.uint16),
        pos_contrast_threshold=pos_contrast_threshold,
        neg_contrast_threshold=neg_contrast_threshold,
        refractory_period=refractory_period,
        bayer_pattern=bayer_pattern
    )

    # linearly interpolate event camera poses at the image timestamps
    distorted_images_path = os.path.join(
        non_raw_events_path,
        DISTORTED_IMAGES_FOLDER_NAME_FORMAT_STR.format(args.camera_position)
    )
    image_timestamps_path = os.path.join(
        distorted_images_path,
        IMAGE_TIMESTAMPS_FILENAME_FORMAT_STR.format(args.camera_position)
    )
    image_timestamp = np.loadtxt(image_timestamps_path)                         # (I)
    image_timestamp = US_TO_NS * image_timestamp                                # (I)
    image_timestamp = image_timestamp.astype(np.int64)
    image_timestamp = image_timestamp - init_T_wc_timestamp

    is_valid_image = (0 <= image_timestamp) \
                     & (image_timestamp <= T_wc_timestamp[-1])                  # [I]
    is_valid_image[:TRIM_INITIAL_NUM_IMAGES] = False
    image_timestamp = image_timestamp[is_valid_image]                           # [I']

    event_trajectory = Trajectory(
        t=torch.from_numpy(T_wc_timestamp),
        position=torch.from_numpy(T_wc_position),
        orientation=torch.from_numpy(T_wc_orientation),
        orientation_is_xyzw=True,
    )
    with torch.no_grad():
        T_w_event_pose = event_trajectory.get_pose_at(                         # (I', 7)
            torch.from_numpy(image_timestamp), method=InterpMethod.SLERP
        )
        T_w_event_position = T_w_event_pose[:, :3]                             # (I', 3)
        T_w_event_orientation = quaternion_to_matrix(T_w_event_pose[:, 3:7])   # (I', 3, 3)

    # derive the RGB camera poses from the event camera poses
    T_w_event = np.zeros((len(T_w_event_position), 4, 4), dtype=np.float32)     # (I', 4, 4)
    T_w_event[:, :3, 3] = T_w_event_position.numpy()
    T_w_event[:, :3, :3] = T_w_event_orientation.numpy()
    T_w_event[:, 3, 3] = 1

    T_imu_rgb = camera_calibration.T_imu_cam[rgb_cam_idx]
    T_imu_rgb = se3_json_to_mat(T_imu_rgb)                                      # (4, 4)
    T_event_rgb = np.linalg.inv(T_imu_event) @ T_imu_rgb                        # (4, 4)
    T_w_rgb = T_w_event @ T_event_rgb                                           # (I', 4, 4)

    # convert the RGB camera poses from a common to the OpenGL convention
    T_w_rgb = T_w_rgb @ T_CCOMMON_COPENGL                                       # (I', 4, 4)


    posed_undistorted_images_path = os.path.join(
        args.preprocessed_dataset_path, POSED_UNDISTORTED_IMAGES_FOLDER_NAME
    )
    transforms_path = os.path.join(
        posed_undistorted_images_path,
        STAGE_TRANSFORMS_FILENAME_FORMAT_STR.format(STAGE)
    )
    valid_image_index = is_valid_image.nonzero()[0]                             # (I')
    image_filename = list(                                                      # (I')
        map(IMAGE_FILENAME_FORMAT_STR.format, valid_image_index)
    )
    transforms = {
        "intrinsics": new_rgb_intrinsics.tolist(),
        "frames": [ { "file_path": os.path.join(
                                    ".", STAGE, os.path.splitext(filename)[0]
                                   ),
                      "transform_matrix": tf_matrix.tolist() }
                    for filename, tf_matrix in zip(image_filename, T_w_rgb) ]
    }
    os.mkdir(posed_undistorted_images_path)
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=4)

    # undistort the RGB images & save them
    stage_undistorted_images_path = os.path.join(
        posed_undistorted_images_path, STAGE
    )
    os.mkdir(stage_undistorted_images_path)
    for filename in image_filename:
        distorted_image_path = os.path.join(distorted_images_path, filename)
        distorted_image = cv2.imread(
            distorted_image_path, cv2.IMREAD_UNCHANGED
        )
        undistorted_image = cv2.fisheye.undistortImage(
            distorted_image, rgb_intrinsics, rgb_distortion_params,
            Knew=new_rgb_intrinsics
        )
        undistorted_image_path = os.path.join(
            stage_undistorted_images_path, filename
        )
        cv2.imwrite(undistorted_image_path, undistorted_image)


def se3_json_to_mat(se3_json):
    se3_vector = np.array(                                                      # (7)
        [ se3_json.px, se3_json.py, se3_json.pz,
          se3_json.qx, se3_json.qy, se3_json.qz, se3_json.qw ],
        dtype=np.float32
    )
    se3_matrix = se3_vec_to_mat(se3_vector)                                     # (4, 4)
    return se3_matrix


def se3_vec_to_mat(se3_vector):                                                 # ([N,] 7)
    assert se3_vector.ndim in (1, 2)
    assert se3_vector.shape[-1] == 7

    position = se3_vector[..., :3]                                              # ([N,] 3)
    orientation = se3_vector[..., 3:]                                           # ([N,] 4)

    if se3_vector.ndim == 1:
        se3_matrix = np.zeros((4, 4), dtype=position.dtype)                     # (4, 4)
    else:   # elif position.ndim == 2:
        N = se3_vector.shape[0]
        se3_matrix = np.zeros((N, 4, 4), dtype=position.dtype)                  # (N, 4, 4)

    se3_matrix[..., :3, 3] = position
    orientation = Rotation.from_quat(orientation)
    se3_matrix[..., :3, :3] = orientation.as_matrix()                           
    se3_matrix[..., 3, 3] = 1

    return se3_matrix


def se3_mat_to_vec(se3_matrix):                                                 # ([N,] 4, 4)
    assert se3_matrix.ndim in (2, 3)
    assert se3_matrix.shape[-2:] == (4, 4)
    assert np.all(se3_matrix[..., 3, :] == np.array([0, 0, 0, 1]))

    position = se3_matrix[..., :3, 3]                                           # ([N,] 3)
    orientation = Rotation.from_matrix(se3_matrix[..., :3, :3])
    orientation = orientation.as_quat().astype(np.float32)                      # ([N,] 4)
    se3_vector = np.concatenate(( position, orientation ), axis=-1)             # ([N,] 7)

    return se3_vector


def filter_event(
    event_position,
    event_timestamp,
    event_polarity,
    T_wc_timestamp
):
    valid_indices = (T_wc_timestamp[0] <= event_timestamp) \
                    & (event_timestamp <= T_wc_timestamp[-1])
    event_position = event_position[valid_indices, :].copy(order="C")
    event_timestamp = event_timestamp[valid_indices].copy(order="C")
    event_polarity = event_polarity[valid_indices].copy(order="C")

    return event_position, event_timestamp, event_polarity


def undistort_event_positions(
    event_x,
    event_y,
    intrinsics,
    distortion,
    target_intrinsics=None,
):
    if event_x.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    K = intrinsics.astype(np.float64, copy=False)
    D = distortion.astype(np.float64, copy=False)
    points = np.stack((event_x, event_y), axis=-1).astype(np.float64, copy=False)
    points = points.reshape(-1, 1, 2)
    if target_intrinsics is None:
        target_intrinsics = K
    P = target_intrinsics.astype(np.float64, copy=False)
    undistorted = cv2.fisheye.undistortPoints(
        points,
        K,
        D,
        P=P
    )
    undistorted = undistorted.reshape(-1, 2).astype(np.float32)
    return undistorted


def compute_in_bounds_mask(points, width, height):
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    x = points[:, 0]
    y = points[:, 1]
    return (
        (0.0 <= x)
        & (x < float(width))
        & (0.0 <= y)
        & (y < float(height))
    )


def remove_small_black_dots(mask):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MASK_MORPH_KERNEL_SIZE, MASK_MORPH_KERNEL_SIZE),
    )
    dilated = cv2.dilate(mask, kernel, iterations=MASK_MORPH_ITERATIONS)
    return cv2.erode(dilated, kernel, iterations=MASK_MORPH_ITERATIONS)


def sanitize_sequence_name(sequence_name):
    return sequence_name.replace("/", "_")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Script for converting TUM-VIE datasets to"
                     " pre-processed ESIM format.")
    )
    parser.add_argument(
        "sequence_name", type=str,
        help=("Desired TUM-VIE dataset sequence for conversion.")
    )
    parser.add_argument(
        "raw_dataset_path", type=str,
        help="Path to the raw TUM-VIE datasets folder."
    )
    parser.add_argument(
        "preprocessed_dataset_path", type=str,
        help="Desired path to the pre-processed TUM-VIE dataset."
    )
    parser.add_argument(
        "--camera_position", type=str, choices=CAMERA_POSITIONS,
        default="left", help="Left or right event & RGB camera to convert."
    )
    parser.add_argument(
        "--start_timestamp", type=int, default=0,
        help="Trim the sequence to start at the given timestamp (inclusive)."
    )
    parser.add_argument(
        "--end_timestamp", type=int, default=float("inf"),
        help="Trim the sequence to end at the given timestamp (exclusive)."
    )
    parser.add_argument(
        "--event_chunk_size", type=int, default=5_000_000,
        help=("Number of events to process per chunk when streaming from the"
              " HDF5 event file.")
    )
    parser.add_argument(
        "--undistorted_img_width", type=int, default=1024,
        help=("Optional target width for undistorted events/camera calibration."
              " Defaults to the raw event camera width.")
    )
    parser.add_argument(
        "--undistorted_img_height", type=int, default=1024,
        help=("Optional target height for undistorted events/camera calibration."
              " Defaults to the raw event camera height.")
    )
    args = parser.parse_args()

    main(args)
