# Computer Vision projects and examples

This repository contains multiple implemented Computer Vision projects as examples and guidance.

### Dependencies

The repository and developed features depend highly in the following packages:
- PyTorch
- OpenCV
- Ultralytics
- Supervision

## 🖥️ Installation

### Clone repository

```bash
git clone https://github.com/xaristeidou/ComputerVision.git
```

### Virtual Environment (Optional, but recommended)

Create a new virtual environment. Recommended `Python>=3.9` version (It may work with older version, but not tested).

```bash
cd ComputerVision
python3 -m venv computer_vision_venv
source computer_vision_venv/bin/activate
pip install --upgrade pip
```

### Install packages

#### PyTorch installation (skip if installed already)

It is recommended to install PyTorch before running requirements installation, especially if you want to download PyTorch with specific CUDA version or for specific OS. More details in the official link [PyTorch installation](https://pytorch.org/get-started/locally/).

#### Install libraries

```bash
pip install -r requirements.txt
```
Required libraries are used with specific/frozen versions and are continuously updated after testing of new released versions.

<br>

## 🛣️🚗 Vehicles counting for multiple lanes

Vehicle counting for total cars passed from a direction and reverse using line zone, and individual counting for each road lane using polygon zones.

<!-- Placeholder for image of project -->

```bash
cd vehicle_lane_counting
python3 vehicle_lane_counting.py
```

By default an asset video from `supervision` package is used. Multiple parameters like the video source path, polygon zones coordinates/color, etc., can be modified from the `config.yaml` file in `vehicle_lane_counting` folder.

To use a custom video just paste the path in `video_path` parameter and set `is_supervision_asset` to `False`. Example provided as follows:
