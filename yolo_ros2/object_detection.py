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

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )

    def image_callback(self, img_msg):
        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().warn(str(e))

        detection_results = self.detection_model(img)
        annotated_frame = detection_results[0].plot()

        cv2.imshow('result', annotated_frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    object_detection_node = ObjectDetection()
    try:
        rclpy.spin(object_detection_node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()