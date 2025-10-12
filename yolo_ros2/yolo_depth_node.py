#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import cv2

class ImageDepthSyncNode(Node):
    def __init__(self):
        super().__init__('image_depth_sync_node')
        self.bridge = CvBridge()

        # --- サブスクライバを作成 ---
        self.color_sub = Subscriber(self, Image, '/camera/color/image_raw')
        self.color_info_sub = Subscriber(self, CameraInfo, '/camera/color/camera_info')
        #self.depth_sub = Subscriber(self, Image, '/camera/depth/image_rect_raw')
        #self.depth_info_sub = Subscriber(self, CameraInfo, '/camera/depth/camera_info')

        # --- 同期器を作成 ---
        # ApproximateTimeSynchronizer( [subscribers], queue_size, slop)
        # slop は許容する時間ずれ（秒）
        self.ts = ApproximateTimeSynchronizer(
            [self.color_sub, self.color_info_sub],
            queue_size=50,
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info("ImageDepthSyncNode initialized and waiting for messages...")

    def sync_callback(self, color_msg, color_info, depth_msg, depth_info):
        # --- OpenCV 形式に変換 ---
        color_img = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        #depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        # --- 深度画像の可視化 ---
        #depth_vis = cv2.convertScaleAbs(depth_img, alpha=0.03)
        #depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # --- 表示 ---
        cv2.imshow("Color", color_img)
        #cv2.imshow("Depth", depth_vis)
        cv2.waitKey(1)

        # --- デバッグ出力 ---
        self.get_logger().info(
            f"Received synced frames: color {color_img.shape}, depth {depth_img.shape}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = ImageDepthSyncNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
