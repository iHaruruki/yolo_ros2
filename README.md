# yolo_ros2
[![ROS 2 Distro - Humble](https://img.shields.io/badge/ros2-Humble-blue)](https://docs.ros.org/en/humble/)

## 🚀 Overview

## 📦 Features

## 🛠️ Setup
Install ros2 packages
```bash
sudo apt install -y ros-$ROS_DISTRO-vision-msgs ros-$ROS_DISTRO-message-filters ros-$ROS_DISTRO-cv-bridge
```
Install YOLOv8
```bash
pip3 install ultralytics
pip3 uninstall -y opencv-python
```
> [!NOTE]
> open-python は ultralytics とともに自動的にインストールされます．したがって,open-contrib-python との競合を避けるためにこれを削除する必要があります．

## 🎮 Usage
### Object detection
Run camera
```bash
ros2 launch orbbec_camera astra_stereo_u3.launch.py
```
```bash
ros2 launch astra_camera astra_pro.launch.xml
```
Run `object_detection_node`
```bash
ros2 run yolo_ros2 object_detection_node --ros-args --remap image_raw:=/camera/color/image_raw
```
### Object segmentation
Run camera
```bash
ros2 launch orbbec_camera astra_stereo_u3.launch.py
```
```bash
ros2 launch astra_camera astra_pro.launch.xml
```
Run `object_segmentation_node`
```bash
ros2 run yolo_ros2 object_segmentation_node --ros-args --remap image_raw:=/camera/color/image_raw
```
### Object detection tf
Run camera
```bash
ros2 launch orbbec_camera astra_stereo_u3.launch.py
```
```bash
ros2 launch astra_camera astra_pro.launch.xml
```
Run `object_detection_tf_node`
```bash
ros2 run yolo_ros2 object_detection_tf_node
```

## 👤 Authors
- **[iHaruruki](https://github.com/iHaruruki)** — Main author & maintainer

## 📚 References
- [YOLO on ROS](https://docs.ultralytics.com/ja/guides/ros-quickstart/#point-clouds-step-by-step-usage)
- [ROS 2とPythonで作って学ぶAIロボット入門 改訂第2版のサポートサイト](https://github.com/AI-Robot-Book-Humble/chapter5/tree/main/yolov8_ros2)
