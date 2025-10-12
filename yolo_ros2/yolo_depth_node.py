#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from vision_msgs.msg import Detection2D, Detection2DArray, BoundingBox2D, ObjectHypothesisWithPose
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import cv2
import tf2_ros

from message_filters import Subscriber, ApproximateTimeSynchronizer

# Ultralytics YOLOv11n を使用
# pip install ultralytics
from ultralytics import YOLO

class YoloCenterDistanceNode(Node):
    def __init__(self):
        super().__init__('yolo_center_distance_node')

        # ========= パラメータ =========
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('depth_info_topic', '/camera/depth/camera_info')
        self.declare_parameter('model_path', './models/yolo11n.pt')  # 例: ./yolo11n.pt
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('iou', 0.45)
        self.declare_parameter('device', 'cpu')  # 'cpu' or '0' (GPU)
        self.declare_parameter('publish_overlay', True)
        self.declare_parameter('center_window', 5)  # 中心近傍のウィンドウ（奇数）

        rgb_topic       = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic     = self.get_parameter('depth_topic').get_parameter_value().string_value
        depth_info_topic= self.get_parameter('depth_info_topic').get_parameter_value().string_value
        model_path      = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf       = float(self.get_parameter('conf').value)
        self.iou        = float(self.get_parameter('iou').value)
        self.device     = self.get_parameter('device').get_parameter_value().string_value
        self.pub_overlay= bool(self.get_parameter('publish_overlay').value)
        self.win        = int(self.get_parameter('center_window').value)
        self.win        = self.win if self.win % 2 == 1 else self.win + 1  # 奇数に

        # ========= YOLO 読み込み =========
        try:
            self.model = YOLO(model_path)
            # Ultralytics の task/type は自動判別。conf, iou は predict 引数で渡す
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            raise

        # ========= 同期購読 =========
        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None

        self.rgb_sub   = Subscriber(self, Image, rgb_topic)
        self.depth_sub = Subscriber(self, Image, depth_topic)
        self.info_sub  = Subscriber(self, CameraInfo, depth_info_topic)

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub],
            queue_size=10,
            slop=0.05
        )
        self.sync.registerCallback(self.synced_callback)

        # ========= Publisher =========
        self.pub_det   = self.create_publisher(Detection2DArray, '/yolo/detections', 10)
        self.pub_arr   = self.create_publisher(Float32MultiArray, '/yolo/center_depths', 10)
        if self.pub_overlay:
            self.pub_img = self.create_publisher(Image, '/yolo/overlay', 10)

        self.get_logger().info('YOLO Center Distance node started.')

        # ========= TF Broadcaster =========
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.parent_frame = "camera_link"  # カメラTF名。必要に応じて変更

    # CameraInfo から内部パラメータを確定
    def _update_intrinsics(self, info: CameraInfo):
        self.fx = info.k[0]
        self.fy = info.k[4]
        self.cx = info.k[2]
        self.cy = info.k[5]

    def _depth_to_meters(self, depth_cv, encoding: str):
        # 32FC1: そのまま[m]、16UC1: [mm] を [m] に変換
        if encoding.lower() in ['32fc1', '32fc']:
            return depth_cv.astype(np.float32)
        elif encoding.lower() in ['16uc1', '16uc']:
            return depth_cv.astype(np.float32) / 1000.0
        else:
            # 不明形式は最善努力で float に
            return depth_cv.astype(np.float32)

    def _median_center_depth(self, depth_m: np.ndarray, cx: float, cy: float):
        h, w = depth_m.shape[:2]
        x = int(round(cx))
        y = int(round(cy))
        half = self.win // 2
        x0, x1 = max(0, x - half), min(w, x + half + 1)
        y0, y1 = max(0, y - half), min(h, y + half + 1)
        roi = depth_m[y0:y1, x0:x1]
        valid = roi[np.isfinite(roi)]
        valid = valid[(valid > 0.05) & (valid < 20.0)]  # 5cm〜20m の範囲
        if valid.size == 0:
            return float('nan')
        return float(np.median(valid))

    def _pixel_to_camera(self, u: float, v: float, Z: float):
        if self.fx is None:
            return (float('nan'), float('nan'), Z)
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        return (float(X), float(Y), float(Z))

    def synced_callback(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        # Intrinsics 更新
        if self.fx is None:
            self._update_intrinsics(info_msg)

        # 画像変換
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg)
        depth_m = self._depth_to_meters(depth, depth_msg.encoding)

        # 推論
        try:
            results = self.model.predict(
                source=rgb,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False
            )
        except Exception as e:
            self.get_logger().error(f'Inference error: {e}')
            return

        det_arr = Detection2DArray()
        det_arr.header = rgb_msg.header

        flat = []  # Float32MultiArray のデータ

        overlay = rgb.copy()

        # Ultralytics の出力解釈
        # results[0].boxes.xyxy (N,4), .conf (N), .cls (N)
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls  = boxes.cls.cpu().numpy().astype(int)

            for i in range(xyxy.shape[0]):
                x1, y1, x2, y2 = xyxy[i]
                score = float(conf[i])
                cid   = int(cls[i])

                # bbox 中心
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)

                # 中心距離（メディアン）
                Z = self._median_center_depth(depth_m, cx, cy)
                X, Y, Z = self._pixel_to_camera(cx, cy, Z)

                # ---- vision_msgs/Detection2D を詰める ----
                det = Detection2D()
                det.header = rgb_msg.header

                bbox = BoundingBox2D()
                # Use the existing sub-message instance to avoid type assertion errors
                # BoundingBox2D.center is a Pose2D message which contains a 'position' (Point) and 'theta'
                bbox.center.position.x = float((x1 + x2) / 2.0)
                bbox.center.position.y = float((y1 + y2) / 2.0)
                bbox.center.theta = 0.0
                bbox.size_x = float(max(0.0, x2 - x1))
                bbox.size_y = float(max(0.0, y2 - y1))
                det.bbox = bbox

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(cid)
                hyp.hypothesis.score = score
                # 距離は備考的に pose の position.z に格納（2D検出なので便宜的）
                hyp.pose.pose.position.x = X
                hyp.pose.pose.position.y = Y
                hyp.pose.pose.position.z = Z
                det.results.append(hyp)

                det_arr.detections.append(det)

                # ---- Float32MultiArray に [cx, cy, Z, class_id, score] で詰める ----
                flat.extend([float(cx), float(cy), float(Z), float(cid), float(score)])

                # ---- 可視化 ----
                if self.pub_overlay:
                    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f'id:{cid} {score:.2f} Z:{Z:.2f}m'
                    cv2.putText(overlay, label, (int(x1), max(0, int(y1)-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    cv2.circle(overlay, (int(cx), int(cy)), 4, (255, 0, 0), -1)

                # ---- TF 配信 ----
                if not np.isnan(Z):
                    t = TransformStamped()
                    t.header.stamp = rgb_msg.header.stamp
                    t.header.frame_id = self.parent_frame
                    t.child_frame_id = f"object_{cid}_{i}"

                    t.transform.translation.x = X
                    t.transform.translation.y = Y
                    t.transform.translation.z = Z
                    t.transform.rotation.x = 0.0
                    t.transform.rotation.y = 0.0
                    t.transform.rotation.z = 0.0
                    t.transform.rotation.w = 1.0

                    self.tf_broadcaster.sendTransform(t)

        # Publish detections
        self.pub_det.publish(det_arr)

        # Publish array
        arr = Float32MultiArray()
        arr.layout.dim.append(MultiArrayDimension(label='rows', size=len(flat)//5, stride=len(flat)))
        arr.layout.dim.append(MultiArrayDimension(label='cols', size=5, stride=5))
        arr.data = flat
        self.pub_arr.publish(arr)

        # Publish overlay
        if self.pub_overlay:
            img_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            img_msg.header = rgb_msg.header
            self.pub_img.publish(img_msg)


def main():
    rclpy.init()
    node = YoloCenterDistanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
