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
        self.declare_parameter('model_path', '~/ros2_ws/src/yolo_ros2/yolo_models/yolo26s-seg.pt')
        self.declare_parameter('roi_scale', 0.5)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('depth_median_window', 20)
        self.declare_parameter('enable_visualization', True)

        # Get parameters
        self.target_name = self.get_parameter('target_name').value
        self.frame_id = self.get_parameter('frame_id').value
        self.parent_frame_id = self.get_parameter('parent_frame_id').value
        model_path = self.get_parameter('model_path').value
        self.roi_scale = self.get_parameter('roi_scale').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.depth_median_window = self.get_parameter('depth_median_window').value
        self.enable_visualization = self.get_parameter('enable_visualization').value

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
            f'  confidence_threshold: {self.confidence_threshold}\n'
            f'  depth_median_window: {self.depth_median_window}'
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

        # Load YOLO segmentation model
        try:
            if not os.path.exists(model_path_home):
                self.get_logger().warn(f'Model not found at {model_path_home}, attempting auto-download...')
            
            self.detection_model = YOLO(model_path_home)
            self.get_logger().info(f'✓ Segmentation model loaded successfully: {model_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load model {model_path_home}: {str(e)}')
            raise

        self.get_logger().info(f'Object detection initialized for target: {self.target_name}')

    def on_parameter_changed(self, params):
        """Callback for parameter changes at runtime"""
        from rcl_interfaces.msg import SetParametersResult
        
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
            elif param.name == 'enable_visualization':
                self.enable_visualization = param.value
                self.get_logger().info(f'Updated enable_visualization to: {self.enable_visualization}')
        
        return SetParametersResult(successful=True)

    def images_callback(self, msg_info, msg_color, msg_depth):
        try:
            img_color = CvBridge().imgmsg_to_cv2(msg_color, 'bgr8')
            img_depth = CvBridge().imgmsg_to_cv2(msg_depth, 'passthrough')
        except CvBridgeError as e:
            self.get_logger().warn(f'CvBridge Error: {str(e)}')
            return

        if img_color.shape[0:2] != img_depth.shape[0:2]:
            self.get_logger().warn('Color and depth image sizes do not match')
            return

        if img_depth.dtype == np.uint16:
            depth_scale = 1e-3
            img_depth_conversion = True
        elif img_depth.dtype == np.float32:
            depth_scale = 1
            img_depth_conversion = False
        else:
            self.get_logger().warn(f'Unsupported depth image type: {img_depth.dtype}')
            return
        
        # ========== YOLO SEGMENTATION ==========
        # Run segmentation inference
        try:
            results = self.detection_model(img_color, verbose=False, conf=self.confidence_threshold)
        except Exception as e:
            self.get_logger().error(f'YOLO inference error: {str(e)}')
            return
        
        if results is None or len(results) == 0:
            self.get_logger().debug('No detections found')
            return
        
        result = results[0]
        names = result.names
        boxes = result.boxes
        masks = result.masks  # Segmentation masks
        
        # Plot annotated frame with segmentation
        annotated_frame = result.plot()
        
        if self.enable_visualization:
            cv2.imshow('YOLO Segmentation', annotated_frame)
            cv2.waitKey(1)
        
        # ========== FIND TARGET OBJECT ==========
        box = None
        mask = None
        
        if boxes is not None and masks is not None:
            for b, m in zip(boxes, masks):
                class_id = int(b.cls[0])
                class_name = names[class_id]
                confidence = float(b.conf[0])
                
                if class_name == self.target_name and confidence >= self.confidence_threshold:
                    box = b
                    mask = m
                    self.get_logger().debug(
                        f'Found target: {class_name} (confidence: {confidence:.2f})')
                    break
        
        # ========== EXTRACT 3D POSITION ==========
        depth = 0
        (bu1, bu2, bv1, bv2) = (0, 0, 0, 0)
        
        if box is not None:
            # Extract bounding box coordinates
            bu1, bv1, bu2, bv2 = [int(i) for i in box.xyxy.cpu().numpy()[0]]
            
            # Calculate ROI (Region of Interest) with roi_scale parameter
            a = self.roi_scale
            roi_u1 = round((bu1 + bu2) / 2 - (bu2 - bu1) * a / 2)
            roi_u2 = round((bu1 + bu2) / 2 + (bu2 - bu1) * a / 2)
            roi_v1 = round((bv1 + bv2) / 2 - (bv2 - bv1) * a / 2)
            roi_v2 = round((bv1 + bv2) / 2 + (bv2 - bv1) * a / 2)  # FIXED: was (bv2 - bv2)
            
            # Bounds checking
            roi_u1 = max(0, roi_u1)
            roi_u2 = min(img_depth.shape[1] - 1, roi_u2)
            roi_v1 = max(0, roi_v1)
            roi_v2 = min(img_depth.shape[0] - 1, roi_v2)
            
            # Center pixel coordinates
            u = round((bu1 + bu2) / 2)
            v = round((bv1 + bv2) / 2)
            
            # Extract depth using median filter for noise reduction
            depth_roi = img_depth[roi_v1:roi_v2+1, roi_u1:roi_u2+1]
            if depth_roi.size > 0:
                # Filter out zero values (invalid depth)
                valid_depths = depth_roi[depth_roi > 0]
                if len(valid_depths) > 0:
                    depth = np.median(valid_depths)
                else:
                    self.get_logger().debug('No valid depth values in ROI')
            
            # Calculate 3D coordinates if depth is valid
            if depth > 0:
                z = float(depth) * depth_scale
                fx = msg_info.k[0]
                fy = msg_info.k[4]
                cx = msg_info.k[2]
                cy = msg_info.k[5]
                x = z / fx * (u - cx)
                y = z / fy * (v - cy)
                
                self.get_logger().debug(
                    f'{self.target_name} Position: ({x:.3f}m, {y:.3f}m, {z:.3f}m)')
                
                # ========== BROADCAST TF TRANSFORM ==========
                ts = TransformStamped()
                ts.header.stamp = self.get_clock().now().to_msg()
                ts.header.frame_id = self.parent_frame_id
                ts.child_frame_id = self.frame_id
                ts.transform.translation.x = x
                ts.transform.translation.y = y
                ts.transform.translation.z = z
                ts.transform.rotation.w = 1.0  # No rotation
                self.broadcaster.sendTransform(ts)

                # ========== PUBLISH POINT MESSAGE ==========
                point_msg = PointStamped()
                point_msg.header = msg_depth.header
                point_msg.point.x = x
                point_msg.point.y = y
                point_msg.point.z = z
                self.point_pub.publish(point_msg)
            else:
                self.get_logger().debug(
                    f'Invalid depth value for {self.target_name} at ROI [{roi_u1}:{roi_u2}, {roi_v1}:{roi_v2}]')

        # ========== VISUALIZE DEPTH IMAGE ==========
        if self.enable_visualization:
            img_depth_vis = img_depth.copy()
            
            # Scale depth image for visualization
            if img_depth_conversion and img_depth_vis.dtype == np.uint16:
                img_depth_vis = img_depth_vis.astype(np.uint16) * 16
            
            # Draw bounding box on depth image if object detected
            if depth > 0:
                pt1 = (int(bu1), int(bv1))
                pt2 = (int(bu2), int(bv2))
                cv2.rectangle(img_depth_vis, pt1=pt1, pt2=pt2, color=0xffff, thickness=2)
                
                # Draw ROI box
                roi_pt1 = (int(roi_u1), int(roi_v1))
                roi_pt2 = (int(roi_u2), int(roi_v2))
                cv2.rectangle(img_depth_vis, pt1=roi_pt1, pt2=roi_pt2, color=0x7fff, thickness=1)
            
            cv2.imshow('Depth Image', img_depth_vis)
            cv2.waitKey(1)


def main():
    rclpy.init()
    try:
        object_detection_tf = ObjectDetectionTF()
        rclpy.spin(object_detection_tf)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Fatal Error: {e}')
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()