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
- [Licence plate blurring for data privacy](#-license-plate-blurring)

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

<br>

## 🚗🔲🔒 License Plate blurring

👉 Detection and blur of license plate of cars. This process can be used for privacy reasons and data protection.

![License Plate blurring](/assets/images/license_plate_blurring.png)

```bash
cd license_plate_blurring
python3 license_plate_blurring.py
```

Because license plates are not included in the COCO dataset, the model that is used is trained with custom dataset. By default, YOLO model architectures are supported. If needed, change the model architecture in `license_plate_blurring.py` file.

⚠️ Notice that the used model is a small and fast model architecture, just for presentation purposes. If you want to use a more accurate model, you should train it to more robust and large license plate dataset.

To use a different detection model, paste the model name in `license_plate_blurring` folder or paste the absolute path in `detection_model`.

To use a custom video just paste the path in `video_source` parameter in `config.yaml`. Example provided as follows:
```
video_source: "/path/to/folder/video.mp4"
```

To use a connected camera real-time image feed provide the number of the connected port. For example, if camera is connected to port `0`:
```
video_source: 0
```