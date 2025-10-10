#!/usr/bin/env python3
import os
import math
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
from geometry_msgs.msg import PoseWithCovariance, Pose, Point, Quaternion
from std_msgs.msg import Header

from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

# Lazy import ultralytics to allow running without it for linting
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def depth_to_meters(depth_array: np.ndarray, encoding: str, depth_scale: float) -> np.ndarray:
    # Convert depth image to meters based on encoding
    if encoding in ('16UC1', 'mono16'):
        # millimeters -> meters via depth_scale (default 0.001)
        return depth_array.astype(np.float32) * depth_scale
    elif encoding in ('32FC1'):
        # already meters
        return depth_array.astype(np.float32)
    else:
        # Try passthrough guess
        return depth_array.astype(np.float32) * depth_scale


def median_depth_at(depth_m: np.ndarray, u: int, v: int, win: int) -> Optional[float]:
    h, w = depth_m.shape
    half = win // 2
    u0, u1 = max(0, u - half), min(w, u + half + 1)
    v0, v1 = max(0, v - half), min(h, v + half + 1)
    patch = depth_m[v0:v1, u0:u1].reshape(-1)
    patch = patch[np.isfinite(patch)]
    patch = patch[(patch > 0.0) & (patch < 100.0)]  # 0 < Z < 100m guard
    if patch.size == 0:
        return None
    return float(np.median(patch))


def backproject(u: float, v: float, z: float, fx: float, fy: float, cx: float, cy: float) -> Tuple[float, float, float]:
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return x, y, z


class YoloDepthNode(Node):
    def __init__(self):
        super().__init__('yolo_depth_node')

        # Parameters
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_depth_info_topic', '/camera/depth/camera_info')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('model_path', 'yolov11n.pt')
        self.declare_parameter('conf_thres', 0.25)
        self.declare_parameter('iou_thres', 0.45)
        self.declare_parameter('max_det', 100)
        self.declare_parameter('depth_scale', 0.001)  # for 16UC1 mm -> meters
        self.declare_parameter('depth_window', 5)     # window size for median depth
        self.declare_parameter('class_filter', [])    # list of class names or indices to include, empty = all

        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.camera_depth_info_topic = self.get_parameter('camera_depth_info_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf_thres = self.get_parameter('conf_thres').get_parameter_value().double_value
        self.iou_thres = self.get_parameter('iou_thres').get_parameter_value().double_value
        self.max_det = int(self.get_parameter('max_det').get_parameter_value().integer_value)
        self.depth_scale = self.get_parameter('depth_scale').get_parameter_value().double_value
        self.depth_window = int(self.get_parameter('depth_window').get_parameter_value().integer_value)
        self.class_filter_param = self.get_parameter('class_filter').get_parameter_value().string_array_value

        # Initialize YOLO
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed. Please `pip install ultralytics`.")

        self.get_logger().info(f'Loading YOLO model: {self.model_path}')
        t0 = time.time()
        self.model = YOLO(self.model_path)
        # Optional: fuse/optimize
        try:
            self.model.fuse()
        except Exception:
            pass
        self.get_logger().info(f'Model loaded in {time.time() - t0:.2f}s')

        # Class filter processing (support indices or names)
        self.class_names = self.model.model.names if hasattr(self.model, 'model') else self.model.names
        self.allowed_classes = set()
        if self.class_filter_param:
            for token in self.class_filter_param:
                token = token.strip()
                if token.isdigit():
                    self.allowed_classes.add(int(token))
                else:
                    # lookup by name
                    name_to_idx = {v: k for k, v in self.class_names.items()}
                    if token in name_to_idx:
                        self.allowed_classes.add(name_to_idx[token])
                    else:
                        self.get_logger().warn(f'class_filter entry "{token}" not found in model classes; ignoring')

        # QoS for sensors
        sensor_qos = QoSProfile(depth=10)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        sensor_qos.history = HistoryPolicy.KEEP_LAST

        self.bridge = CvBridge()

        # Subscribers with ApproximateTimeSynchronizer
        self.sub_rgb = Subscriber(self, Image, self.rgb_topic, qos_profile=sensor_qos)
        self.sub_depth = Subscriber(self, Image, self.depth_topic, qos_profile=sensor_qos)
        # CameraInfo can be unsynced; we will cache the latest
        self.create_subscription(CameraInfo, self.camera_info_topic, self.on_camera_info, qos_profile=sensor_qos)
        self.camera_info: Optional[CameraInfo] = None

        self.sync = ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.synced_callback)

        # Publisher
        self.pub = self.create_publisher(Detection3DArray, 'detections3d', 10)

        self.get_logger().info('yolo_depth_node initialized.')

    def on_camera_info(self, msg: CameraInfo):
        self.camera_info = msg

    def synced_callback(self, rgb_msg: Image, depth_msg: Image):
        if self.camera_info is None:
            self.get_logger().warn_throttle(5000, 'Waiting for CameraInfo...')
            return

        # Convert images
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge RGB conversion failed: {e}')
            return

        try:
            depth_np = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'cv_bridge Depth conversion failed: {e}')
            return

        depth_m = depth_to_meters(depth_np, depth_msg.encoding, self.depth_scale)

        # Run YOLO inference
        try:
            results = self.model.predict(
                source=cv_rgb,  # numpy array BGR is OK
                verbose=False,
                conf=self.conf_thres,
                iou=self.iou_thres,
                max_det=self.max_det,
                imgsz=None,
                device=None  # auto
            )
        except Exception as e:
            self.get_logger().error(f'YOLO inference failed: {e}')
            return

        if not results:
            return
        res = results[0]

        # Extract intrinsics
        K = np.array(self.camera_info.k, dtype=np.float32).reshape(3, 3)
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

        det_array = Detection3DArray()
        det_array.header = Header()
        det_array.header.stamp = rgb_msg.header.stamp
        det_array.header.frame_id = self.camera_info.header.frame_id or rgb_msg.header.frame_id or 'camera_frame'

        boxes = getattr(res, 'boxes', None)
        if boxes is None or boxes.xyxy is None:
            # No detections
            self.pub.publish(det_array)
            return

        # Bring to CPU numpy
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones((xyxy.shape[0],), dtype=np.float32)
            clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((xyxy.shape[0],), dtype=int)
        except Exception:
            # Some older ultralytics versions
            xyxy = boxes.xyxy
            confs = boxes.conf if boxes.conf is not None else np.ones((xyxy.shape[0],), dtype=np.float32)
            clss = boxes.cls.astype(int) if boxes.cls is not None else np.zeros((xyxy.shape[0],), dtype=int)

        H, W = depth_m.shape[:2]
        win = max(1, self.depth_window)

        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i]
            cx_px = int(round((x1 + x2) / 2.0))
            cy_px = int(round((y1 + y2) / 2.0))
            # Clip to image bounds
            cx_px = max(0, min(W - 1, cx_px))
            cy_px = max(0, min(H - 1, cy_px))

            cls_id = int(clss[i])
            if self.allowed_classes and cls_id not in self.allowed_classes:
                continue

            z = median_depth_at(depth_m, cx_px, cy_px, win)
            if z is None or z <= 0.0 or not math.isfinite(z):
                continue

            x, y, z = backproject(cx_px, cy_px, z, fx, fy, cx, cy)

            # Build Detection3D
            det3d = Detection3D()
            det3d.header = det_array.header

            hyp = ObjectHypothesisWithPose()
            # Set label as class name if available, otherwise id
            class_name = self.class_names.get(cls_id, str(cls_id)) if isinstance(self.class_names, dict) else str(cls_id)
            hyp.hypothesis.class_id = class_name
            hyp.hypothesis.score = float(confs[i])

            pose = Pose()
            pose.position = Point(x=x, y=y, z=z)
            # No orientation estimate; use identity
            pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

            hyp.pose.pose = pose
            # covariance left default zeros
            det3d.results.append(hyp)

            # Optional: store approximate size using bbox projected size at depth (not required)
            det_array.detections.append(det3d)

        self.pub.publish(det_array)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = YoloDepthNode()
        rclpy.spin(node)
    finally:
        rclpy.shutdown()