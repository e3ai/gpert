# Geometric-Photometric Event-based 3D Gaussian Ray Tracing (CVPR 2026 Highlight)
| [Webpage](https://e3ai.github.io/gpert/) | [Paper](https://arxiv.org/abs/2512.18640) | Video (Coming soon) |

This repository is the official implementation of [Geometric-Photometric Event-based 3D Gaussian Ray Tracing](https://arxiv.org/abs/2512.18640) by Kai Kohyama, Yoshimitsu Aoki, Guillermo Gallego, and Shintaro Shiba.

<img src="./assets/teaser_gif_short.gif" width="80%" height="auto" alt="Thumbnail"/>


If you use this work in your research, please cite it:

```bibtex
@InProceedings{Kohyama26cvpr,
  author        = {Kai Kohyama and Yoshimitsu Aoki and Guillermo Gallego and Shintaro Shiba},
  title         = {Geometric-Photometric Event-based 3D Gaussian Ray Tracing},
  booktitle     = {Computer Viion and Pattern Recognition (CVPR)},
  year          = 2026
}
```

## Setup
### Tested environments
- Ubuntu 22.04
- Python 3.11.4
- PyTorch 2.7.1
- CUDA 11.8

### Installation
```
# Dependency: Please install PyTorch first.
pip install -r requirements.txt
pip install slangtorch==1.3.4
```

## Data preparation
### TUM-VIE (real data)
1. Please download the following files from [here](https://cvg.cit.tum.de/data/datasets/visual-inertial-event-dataset) into a common folder:
    - <sequence_name>-events_left.h5
    - <sequence_name>-vi_gt_data.tar.gz
    - camera-calibration{A, B}.json
    - mocap-imu-calibration{A, B}.json
2. Extract the `tar.gz` file.
3. Preprocess the raw data with:

    ```python scripts/tum_vie_to_esim.py <sequence_name> <raw_dataset_path> <preprocessed_dataset_path>```

- Our experiments are performed on the `mocap-1d-trans` and `mocap-desk2` sequences.

### Robust e-NeRF (synthetic data)
1. Please download the datasets from [here](https://huggingface.co/datasets/wengflow/robust-e-nerf#setup).
2. Preprocess the raw data with:

    ```python scripts/preprocess_esim.py <sequence_path>/esim.conf <sequence_path>/esim.bag <sequence_path>```
    - *This preprocessing code requires a ROS environment.

## Execution (Train & Test)
### TUM-VIE data
```
python scripts/run.py --config cfg/tum_vie/desk2.yaml
```

### Robust e-NeRF data
```
python scripts/run.py --config cfg/robust_e_nerf/chair.yaml
```

- If you want to run only testing, please set `train: False`, `test: True`, and `gsinit_method: checkpoint`, then set `gsinit_ckpt_path` to the checkpoint path in the config file.


## Authors
- Kai Kohyama [@FlorPeng](https://github.com/FlorPeng)
- Shintaro Shiba [@shiba24](https://github.com/shiba24)

## Acknowledgements
We appreciate the following repositories for inspiration:
- [3DGRUT](https://github.com/nv-tlabs/3dgrut)
- [robust-e-nerf](https://github.com/wengflow/robust-e-nerf?tab=MIT-1-ov-file)


-------
# Additional Resources

* [Secrets of Event-based Optical Flow (ECCV 2022, T-PAMI 2024)](https://github.com/tub-rip/event_based_optical_flow)
* [event-vision-library](https://github.com/shiba24/event-vision-library)

