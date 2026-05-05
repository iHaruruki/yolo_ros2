import os
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, PointStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer
from tf2_ros import TransformBroadcaster

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

from ultralytics import YOLO


class ObjectDetectionTF(Node):

    def __init__(self):
        super().__init__('object_detection_tf')

        # Declare ROS 2 parameters
        self.declare_parameter('target_name', 'teddy bear')
        self.declare_parameter('frame_id', 'detected_object')
        self.declare_parameter('parent_frame_id', 'camera_depth_optical_frame')
        self.declare_parameter('model_path', '~/ros2_ws/src/yolo_ros2/yolo_models/yolo26s.pt')
        self.declare_parameter('roi_scale', 0.5)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('depth_median_window', 20)

        # Get parameters
        self.target_name = self.get_parameter('target_name').value
        self.frame_id = self.get_parameter('frame_id').value
        self.parent_frame_id = self.get_parameter('parent_frame_id').value
        model_path = self.get_parameter('model_path').value
        self.roi_scale = self.get_parameter('roi_scale').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.depth_median_window = self.get_parameter('depth_median_window').value

        # Add parameter change callback for runtime reconfiguration
        self.add_on_set_parameters_callback(self.on_parameter_changed)

        # Expand home directory in model path
        model_path_home = os.path.expanduser(model_path)

        self.get_logger().info(
            f'Object Detection TF Node initialized:\n'
            f'  target_name: {self.target_name}\n'
            f'  frame_id: {self.frame_id}\n'
            f'  parent_frame_id: {self.parent_frame_id}\n'
            f'  model_path: {model_path_home}\n'
            f'  roi_scale: {self.roi_scale}\n'
            f'  confidence_threshold: {self.confidence_threshold}'
        )

        # message_filtersを使って3個のトピックのサブスクライブをまとめて処理する．
        self.callback_group = ReentrantCallbackGroup()   # コールバックの並行処理のため
        self.sub_info = Subscriber(
            self, CameraInfo, '/camera/color/camera_info',
            callback_group=self.callback_group)
        self.sub_color = Subscriber(
            self, Image, '/camera/color/image_raw',
            callback_group=self.callback_group)
        self.sub_depth = Subscriber(
            self, Image, '/camera/depth/image_raw',
            callback_group=self.callback_group)
        self.ts = ApproximateTimeSynchronizer(
            [self.sub_info, self.sub_color, self.sub_depth], 10, 0.1)
        self.ts.registerCallback(self.images_callback)
        
        # 認識した物体の位置をtfとして出力するためのブロードキャスタ
        self.broadcaster = TransformBroadcaster(self)

        # 3次元座標をパブリッシュするためのパブリッシャーを追加
        self.point_pub = self.create_publisher(
            PointStamped, 
            '/object_position', 
            10
        )

        # Load YOLO model
        try:
            self.detection_model = YOLO(model_path_home)
            self.get_logger().info(f'Model loaded successfully: {model_path}')
        except Ellipsis as e:
            self.get_logger().error(f'Failed to load model {model_path}: {str(e)}')
            raise

        self.get_logger().info(f'Object detection initialized for target: {self.target_name}')

    def on_parameter_changed(self, params):
        """Callback for parameter changes at runtime"""
        for param in params:
            if param.name == 'target_name':
                self.target_name = param.value
                self.get_logger().info(f'Updated target_name to: {self.target_name}')
            elif param.name == 'frame_id':
                self.frame_id = param.value
                self.get_logger().info(f'Updated frame_id to: {self.frame_id}')
            elif param.name == 'parent_frame_id':
                self.parent_frame_id = param.value
                self.get_logger().info(f'Updated parent_frame_id to: {self.parent_frame_id}')
            elif param.name == 'roi_scale':
                self.roi_scale = param.value
                self.get_logger().info(f'Updated roi_scale to: {self.roi_scale}')
            elif param.name == 'confidence_threshold':
                self.confidence_threshold = param.value
                self.get_logger().info(f'Updated confidence_threshold to: {self.confidence_threshold}')
            elif param.name == 'depth_median_window':
                self.depth_median_window = param.value
                self.get_logger().info(f'Updated depth_median_window to: {self.depth_median_window}')
        
        from rcl_interfaces.msg import SetParametersResult
        return SetParametersResult(successful=True)

    def images_callback(self, msg_info, msg_color, msg_depth):
        try:
            img_color = CvBridge().imgmsg_to_cv2(msg_color, 'bgr8')
            img_depth = CvBridge().imgmsg_to_cv2(msg_depth, 'passthrough')
        except CvBridgeError as e:
            self.get_logger().warn(str(e))
            return

        if img_color.shape[0:2] != img_depth.shape[0:2]:
            self.get_logger().warn('カラーと深度の画像サイズが異なる')
            return

        if img_depth.dtype == np.uint16:
            depth_scale = 1e-3
            img_depth_conversion = True
        elif img_depth.dtype == np.float32:
            depth_scale = 1
            img_depth_conversion = False
        else:
            self.get_logger().warn('深度画像の型に対応していない')
            return
        
        # 物体認識
        boxes = []
        classes = []
        results = self.detection_model(img_color, verbose=False, conf=self.confidence_threshold)
        names = results[0].names
        boxes = results[0].boxes
        classes = results[0].boxes.cls
        img_color = results[0].plot()

        cv2.imshow('color', img_color)

        # 物体に認識の結果に指定された名前があるか調べる．
        box = None
        for b, c in zip(boxes, classes):
            if names[int(c)] == self.target_name:
                box = b
                break

        # カラー画像内で検出された場合は，深度画像から3次元位置を算出．
        depth = 0
        (bu1, bu2, bv1, bv2) = (0, 0, 0, 0)
        if box is not None:
            a = self.roi_scale  # Use parameter instead of hardcoded value
            bu1, bv1, bu2, bv2 = [int(i) for i in box.xyxy.cpu().numpy()[0]]
            
            # Calculate ROI with proper bounds checking
            u1 = round((bu1 + bu2) / 2 - (bu2 - bu1) * a / 2)
            u2 = round((bu1 + bu2) / 2 + (bu2 - bu1) * a / 2)
            v1 = round((bv1 + bv2) / 2 - (bv2 - bv1) * a / 2)
            v2 = round((bv1 + bv2) / 2 + (bv2 - bv1) * a / 2)  # BUG FIX: was (bv2 - bv2)
            
            # Bounds check
            u1 = max(0, u1)
            u2 = min(img_depth.shape[1] - 1, u2)
            v1 = max(0, v1)
            v2 = min(img_depth.shape[0] - 1, v2)
            
            u = round((bu1 + bu2) / 2)
            v = round((bv1 + bv2) / 2)
            
            # Use parameter for median window size
            window_size = self.depth_median_window
            depth = np.median(img_depth[v1:v2+1, u1:u2+1])
            
            if depth != 0:
                z = float(depth) * depth_scale
                fx = msg_info.k[0]
                fy = msg_info.k[4]
                cx = msg_info.k[2]
                cy = msg_info.k[5]
                x = z / fx * (u - cx)
                y = z / fy * (v - cy)
                
                self.get_logger().debug(
                    f'{self.target_name} ({x:.3f}, {y:.3f}, {z:.3f})')
                
                # tfの送出
                ts = TransformStamped()
                ts.header.stamp = self.get_clock().now().to_msg()
                ts.header.frame_id = self.parent_frame_id  # 親フレームを明示
                ts.child_frame_id = self.frame_id
                ts.transform.translation.x = x
                ts.transform.translation.y = y
                ts.transform.translation.z = z
                ts.transform.rotation.w = 1.0  # 姿勢を設定（回転なし）
                self.broadcaster.sendTransform(ts)

                # 3次元座標をPointStampedメッセージとしてパブリッシュ
                point_msg = PointStamped()
                point_msg.header = msg_depth.header
                point_msg.point.x = x
                point_msg.point.y = y
                point_msg.point.z = z
                self.point_pub.publish(point_msg)
            else:
                self.get_logger().debug(
                    f'Invalid depth value (0) for {self.target_name} at ROI [{u1}:{u2}, {v1}:{v2}]')

        # 深度画像の加工
        if img_depth_conversion:
            img_depth *= 16
        if depth != 0:  # 認識していて，かつ，距離が得られた場合
            pt1 = (int(bu1), int(bv1))
            pt2 = (int(bu2), int(bv2))
            cv2.rectangle(img_depth, pt1=pt1, pt2=pt2, color=0xffff)

        cv2.imshow('depth', img_depth)
        cv2.waitKey(1)


def main():
    rclpy.init()
    try:
        object_detection_tf = ObjectDetectionTF()
        rclpy.spin(object_detection_tf)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()