import rclpy
from rclpy.node import Node
from qwen_vl_msgs.srv import QwenVL
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int8, String
import threading
from rclpy.executors import MultiThreadedExecutor
import time

class QwenVLClientNode(Node):
    def __init__(self):
        super().__init__("qwen_vl_client")
        self.image_sub = self.create_subscription(CompressedImage, "/image", self.image_callback, 10)
        self.qwen_client = self.create_client(QwenVL, "/qwen_vl_service")
        self.describe_command_sub = self.create_subscription(Int8, "/describe_command", self.describe_callback, 10)
        self.text_pub = self.create_publisher(String, "/qwen_description", 10)

        self.t0 = None
        self.img_msg = None
        self.lock = threading.Lock()
        
        # 等待服务可用
        while not self.qwen_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待服务可用...')
    
    def image_callback(self, msg):
        self.img_msg = msg

    def describe_callback(self, msg):
        if msg.data == 1:
            
            if self.img_msg is None:
                self.get_logger().warn('没有可用的图像数据')
                return
            current_img = self.img_msg

            # 在新线程中处理服务调用，避免阻塞
            threading.Thread(target=self.call_qwen_service, args=(current_img,)).start()

    def call_qwen_service(self, img_msg):
        self.t0 = time.time()
        request = QwenVL.Request()
        request.image = img_msg

        print("开始调用服务")
        future = self.qwen_client.call_async(request)

        # 使用回调函数处理结果，而不是阻塞等待
        future.add_done_callback(self.service_response_callback)
        

    def service_response_callback(self, future):
        try:
            result = future.result()
            if result is not None:
                text_msg = String()
                text_msg.data = result.text
                self.text_pub.publish(text_msg)
                print("推理成功,生成文本：", result.text)
                print("调用服务的时间：",time.time() - self.t0)
            else:
                self.get_logger().error('服务调用失败：结果为空')
        except Exception as e:
            self.get_logger().error(f'服务调用异常: {e}')

def main():
    rclpy.init()
    
    # 使用多线程执行器
    executor = MultiThreadedExecutor()
    node = QwenVLClientNode()
    
    try:
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
