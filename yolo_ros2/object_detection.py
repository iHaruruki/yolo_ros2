import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError

from ultralytics import YOLO

class ObjectDetection(Node):
    def __init__(self, **args):
        super().__init__('object_detection')

        self.detection_model = YOLO("yolov8m.pt")