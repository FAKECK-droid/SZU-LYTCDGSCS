import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from ai_msgs.msg import PerceptionTargets
from std_msgs.msg import String, Int8
from cv_bridge import CvBridge
import cv2
import threading
import time

class QRCodeDetector(Node):
    def __init__(self):
        super().__init__('qr_code_detector_opencv')
        self.bridge = CvBridge()
        self.state = 0
        self.subscription = self.create_subscription(
            CompressedImage, 
            '/image', 
            self.image_callback, 10)
        self.qr_publisher = self.create_publisher(String, '/qrcode_info', 10)
        self.state_sub = self.create_subscription(Int8, "/state", self.state_callback, 10)
        self.target_sub = self.create_subscription(PerceptionTargets,'/hobot_dnn_detection',self.target_callback,10)

        self.detect = False
        self.lock = threading.Lock()
        self.latest_image = None
        self.last_result = ""
        self.decode_interval = 5
        self.frame_count = 0
        self.detector = cv2.QRCodeDetector()
        self.processing_thread = threading.Thread(target=self.decode_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.get_logger().info("QR Code Detector Node Started")
    
    def target_callback(self,msg):
        targets = msg.targets
        for target in targets:
            if target.type == "qrcode" and self.state == 1 and not self.detect:
                self.detect = True
                # print("检测到二维码")

    def image_callback(self, msg):
        if self.state != 1:
            return
        self.frame_count += 1
        if self.frame_count % self.decode_interval != 0:
            return
        with self.lock:
            self.latest_image = msg

    def decode_worker(self):
        while rclpy.ok():
            msg = None
            with self.lock:
                if self.latest_image is not None:
                    msg = self.latest_image
                    self.latest_image = None
            if msg is not None and self.state == 1 and self.detect:
                try:
                    cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
                    data, points, _ = self.detector.detectAndDecode(cv_image)
                    if data and data != self.last_result:
                        result_msg = String()
                        result_msg.data = data
                        self.qr_publisher.publish(result_msg)
                        self.get_logger().info(f"Detected QR: {data}")
                        self.last_result = data
                except Exception as e:
                    self.get_logger().error(f"Image processing failed: {str(e)}")
            else:
                time.sleep(0.01)

    def state_callback(self, msg):
        self.state = msg.data

def main(args=None):
    rclpy.init(args=args)
    qr_detector = QRCodeDetector()
    try:
        rclpy.spin(qr_detector)
    except KeyboardInterrupt:
        pass
    finally:
        qr_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()