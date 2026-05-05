#!/usr/bin/env python3

import os
import cv2
import yaml
import threading
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class ColorImageCapture(Node):
    def __init__(self):
        super().__init__('color_image_capture_node')

        self.declare_parameter('save_directory', os.path.expanduser('~/camera_data'))
        self.declare_parameter('window_name', 'color_image_capture')
        self.declare_parameter('display_fps', 30.0)

        self.save_directory = self.get_parameter('save_directory').value
        self.window_name = self.get_parameter('window_name').value
        self.display_fps = float(self.get_parameter('display_fps').value)

        self.image_dir = Path(self.save_directory) / 'images'
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f'Save directory: {self.save_directory}')
        self.get_logger().info('Press SPACE in the image window to save. Press q to quit.')

        self.bridge = CvBridge()

        self.latest_image = None
        self.latest_image_msg = None
        self.camera_info = None
        self.lock = threading.Lock()
        self.camera_info_saved = False

        self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        period = 1.0 / self.display_fps if self.display_fps > 0 else 0.03
        self.timer = self.create_timer(period, self.ui_loop)

    def image_callback(self, msg: Image):
        with self.lock:
            try:
                self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.latest_image_msg = msg
            except Exception as e:
                self.get_logger().error(f'Failed to convert image: {e}')

    def camera_info_callback(self, msg: CameraInfo):
        with self.lock:
            self.camera_info = msg
            if not self.camera_info_saved:
                self.save_camera_info(msg)
                self.camera_info_saved = True

    def ui_loop(self):
        # 表示
        with self.lock:
            frame = None if self.latest_image is None else self.latest_image.copy()

        if frame is not None:
            cv2.imshow(self.window_name, frame)

        # キー入力（ウィンドウがフォーカスされている必要あり）
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Space
            self.capture_image()
        elif key in (ord('q'), ord('Q')):
            self.get_logger().info('Quit requested (q pressed).')
            rclpy.shutdown()

    def capture_image(self):
        with self.lock:
            if self.latest_image is None:
                self.get_logger().warn('No image received yet')
                return

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            image_path = self.image_dir / f'image_{timestamp}.png'

            ok = cv2.imwrite(str(image_path), self.latest_image)
            if not ok:
                self.get_logger().error(f'cv2.imwrite failed: {image_path}')
                return

            self.get_logger().info(f'Image saved: {image_path}')

            metadata = {
                'timestamp': timestamp,
                'frame_id': self.latest_image_msg.header.frame_id if self.latest_image_msg else 'unknown',
                'height': int(self.latest_image.shape[0]),
                'width': int(self.latest_image.shape[1]),
                'channels': int(self.latest_image.shape[2]) if len(self.latest_image.shape) > 2 else 1,
            }

            if self.camera_info is not None:
                metadata['distortion_model'] = self.camera_info.distortion_model
                metadata['d'] = [float(x) for x in self.camera_info.d]
                metadata['k'] = [float(x) for x in self.camera_info.k]
                metadata['r'] = [float(x) for x in self.camera_info.r]
                metadata['p'] = [float(x) for x in self.camera_info.p]

            metadata_path = self.image_dir / f'image_{timestamp}_metadata.yaml'
            with open(metadata_path, 'w') as f:
                yaml.safe_dump(metadata, f, sort_keys=False)

    def save_camera_info(self, camera_info: CameraInfo):
        try:
            camera_info_data = {
                'header': {
                    'frame_id': camera_info.header.frame_id,
                    'stamp': {
                        'sec': int(camera_info.header.stamp.sec),
                        'nanosec': int(camera_info.header.stamp.nanosec),
                    },
                },
                'resolution': {
                    'width': int(camera_info.width),
                    'height': int(camera_info.height),
                },
                'distortion_model': camera_info.distortion_model,
                'd': [float(x) for x in camera_info.d],
                'k': [float(x) for x in camera_info.k],
                'r': [float(x) for x in camera_info.r],
                'p': [float(x) for x in camera_info.p],
                'binning': {
                    'x': int(camera_info.binning_x),
                    'y': int(camera_info.binning_y),
                },
                'roi': {
                    'x_offset': int(camera_info.roi.x_offset),
                    'y_offset': int(camera_info.roi.y_offset),
                    'height': int(camera_info.roi.height),
                    'width': int(camera_info.roi.width),
                    'do_rectify': bool(camera_info.roi.do_rectify),
                },
            }

            info_path = Path(self.save_directory) / 'camera_info.yaml'
            with open(info_path, 'w') as f:
                yaml.safe_dump(camera_info_data, f, sort_keys=False)

            self.get_logger().info(f'Camera info saved: {info_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save camera info: {e}')

    def destroy_node(self):
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ColorImageCapture()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()