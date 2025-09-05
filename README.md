# Computer Vision projects and examples

This repository contains a collection of Computer Vision projects, implemented for learning, guidance, and for fun.

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

### 🔗 Projects Catalog
- [Vehicle counting for multiple lanes](#️-vehicles-counting-for-multiple-lanes)

## 🛣️🚗 Vehicles counting for multiple lanes


👉 Vehicle counting for total cars passed from a direction and reverse using line zone, and individual counting for each road lane using polygon zones.

![Vehicle lane counting](/assets/images/vehicle_lane_counting.png)

```bash
cd vehicle_lane_counting
python3 vehicle_lane_counting.py
```

By default an asset video from `supervision` package is used. Multiple parameters like the video source path, polygon zones coordinates/color, etc., can be modified from the `config.yaml` file in `vehicle_lane_counting` folder.

To use a custom video just paste the path in `video_path` parameter and set `is_supervision_asset` to `False`. Example provided as follows:
```
video_source:
  video_path: "/path/to/folder/video.mp4"
  is_supervision_asset: False
```

To use a connected camera real-time image feed provide the number of the connected port. For example, if camera is connected to port `0`:
```
video_source:
  video_path: 0
  is_supervision_asset: False
```
