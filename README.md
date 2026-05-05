# yolo_ros2
[![ROS 2 Distro - Humble](https://img.shields.io/badge/ros2-Humble-blue)](https://docs.ros.org/en/humble/)

## 🚀 Overview
Object recognition by yolo running as a ROS 2 node.

## 📦 Features
* Object detection
* Object segmentation
* Estimating 3D coordinates of an object using a depth camera

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

Clone repository
```bash
cd ~/ros2_ws/src
git clone https://github.com/iHaruruki/yolo_ros2.git
```
Build
```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select yolo_ros2
source install/setup.bash
```

## 🎮 Usage
### Object detection
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
#### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_name` | string | `teddy bear` | Name of the object to detect (must match YOLO class name) |
| `frame_id` | string | `detected_object` | TF frame ID for detected object |
| `parent_frame_id` | string | `camera_depth_optical_frame` | Parent TF frame (camera frame) |
| `model_path` | string | `~/ros2_ws/src/yolo_ros2/yolo_models/yolo26s.pt` | Path to YOLO model file |
| `roi_scale` | double | `0.5` | Controls how much of the detected bounding box is used to extract the depth value from the depth image. (0.0-1.0) |
| `confidence_threshold` | double | `0.5` | YOLO confidence threshold (0.0-1.0) |
| `depth_median_window` | int | `20` | Window size for extracting depth values - takes the median of NxN pixel region around the object center. |

### Object segmentation
Run camera
```bash
ros2 launch orbbec_camera astra_stereo_u3.launch.py
```
```bash
ros2 launch astra_camera astra_pro.launch.xml
```
Run `object_segmentation_tf_node`
```bash
ros2 run yolo_ros2 object_segmentation_tf_node
```
#### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_name` | string | `teddy bear` | Name of the object to detect (must match YOLO class name) |
| `frame_id` | string | `detected_object` | TF frame ID for detected object |
| `parent_frame_id` | string | `camera_depth_optical_frame` | Parent TF frame (camera frame) |
| `model_path` | string | `~/ros2_ws/src/yolo_ros2/yolo_models/yolo26s-seg.pt` | Path to YOLO model file |
| `roi_scale` | double | `0.5` | Controls how much of the detected bounding box is used to extract the depth value from the depth image. (0.0-1.0) |
| `confidence_threshold` | double | `0.5` | YOLO confidence threshold (0.0-1.0) |
| `depth_median_window` | int | `20` | Window size for extracting depth values - takes the median of NxN pixel region around the object center. |
| `enable_visualization` | bool | ``ture | Show OpenCV windows. |

## 👤 Authors
- **[iHaruruki](https://github.com/iHaruruki)** — Main author & maintainer

## 📚 References
- [ultralytics](https://docs.ultralytics.com/)
- [YOLO on ROS](https://docs.ultralytics.com/ja/guides/ros-quickstart/#point-clouds-step-by-step-usage)
- [ROS 2とPythonで作って学ぶAIロボット入門 改訂第2版のサポートサイト](https://github.com/AI-Robot-Book-Humble/chapter5/tree/main/yolov8_ros2)
