# yolo_ros2
## 🛠️ Setup
Install ros2 packages
```bash
sudo apt install ros-humble-vision-msgs
sudo apt install ros-humble-message-filters
sudo apt install ros-humble-cv-bridge
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
ros2 run usb_cam usb_cam_node_exe
```
Run `object_detection_node`
```bash
ros2 run yolo_ros2 object_detection_node
```

## 👤 Authors
- **[iHaruruki](https://github.com/iHaruruki)** — Main author & maintainer

## 📚 References
- [YOLO on ROS](https://docs.ultralytics.com/ja/guides/ros-quickstart/#point-clouds-step-by-step-usage)
- [ROS 2とPythonで作って学ぶAIロボット入門 改訂第2版のサポートサイト](https://github.com/AI-Robot-Book-Humble/chapter5/tree/main/yolov8_ros2)
