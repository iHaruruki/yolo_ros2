import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
from message_filters import Subscriber, ApproximateTimeSynchronizer

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')

        # message_filters を使って Image と CameraInfo を同期
        self.image_sub = Subscriber(self, Image, '/camera/color/image_raw')
        self.info_sub  = Subscriber(self, CameraInfo, '/camera/color/camera_info')
        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.info_sub],
            queue_size=10,
            slop=0.1,
            allow_headerless=False
        )
        self.ts.registerCallback(self.synced_callback)

        # 検出結果の可視化画像を出力するトピック
        self.pub = self.create_publisher(Image, '/yolo/image_detections', 10)

    def synced_callback(self, img_msg: Image, info_msg: CameraInfo):
        # CameraInfo は info_msg.K, info_msg.D などで内部パラメータにアクセス可能
        frame = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        
        # 必要なら、cv2.undistort で歪み補正
        # K = np.array(info_msg.k).reshape(3,3)
        # D = np.array(info_msg.d)
        # frame = cv2.undistort(frame, K, D)

        results = self.model(frame)[0]
        annotated = results.plot()
        out_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        out_msg.header = img_msg.header
        self.pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
