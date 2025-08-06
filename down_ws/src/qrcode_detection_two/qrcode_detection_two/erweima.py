import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Int8
from cv_bridge import CvBridge
import cv2
from pyzbar import pyzbar
import threading
import time

class QRCodeDetector(Node):
    def __init__(self):
        super().__init__('qr_code_detector')
        self.bridge = CvBridge()
        self.state = 0  # 0不开启识别 1开启识别
        self.frame_count = 0
        self.decode_interval = 5  # 每5帧处理一次
        self.last_result = ""
        self.lock = threading.Lock()
        self.latest_image = None

        self.subscription = self.create_subscription(
            CompressedImage,
            '/image',
            self.image_callback,
            10
        )
        self.state_sub = self.create_subscription(
            Int8,
            '/state',
            self.state_callback,
            10
        )
        self.qr_publisher = self.create_publisher(String, '/qrcode_info', 10)

        self.processing_thread = threading.Thread(target=self.decode_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.get_logger().info("QR Code Detector Node Started (Optimized)")

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
            if msg is not None and self.state == 1:
                try:
                    cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
                    # 缩小图片加速
                    # cv_image = cv2.resize(cv_image, (0, 0), fx=0.7, fy=0.7)
                    barcodes = pyzbar.decode(cv_image)
                    detected_data = [barcode.data.decode("utf-8") for barcode in barcodes]
                    result = "|".join(detected_data)
                    if barcodes:  
                        barcode = barcodes[0]  
                        data = barcode.data.decode("utf-8")
    
                        if data.isdigit():
                            if int(data) % 2 != 0 :
                                direction = "顺时针" 
                            else :
                                direction = "逆时针"
                            result = f"{data}{direction}"  
                        else:
                            result = data  
                    else:
                        result = ""  
                    if result:
                        result_msg = String()
                        result_msg.data = result
                        self.qr_publisher.publish(result_msg)
                        self.get_logger().info(f"Detected QR: {result}")
                        self.last_result = result
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