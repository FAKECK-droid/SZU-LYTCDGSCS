import cv2
import rclpy
from rclpy.node import Node 
import cv_bridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray  # 注意：发布端用的是Float32MultiArray，不是Float64MultiArray
import numpy as np

class VisualizationNode(Node):
    def __init__(self):
        super().__init__("visualization_node")
        self.img_sub = self.create_subscription(CompressedImage,"/image",self.img_callback,10)
        # 修改：使用Float32MultiArray匹配发布端
        self.visualization_sub = self.create_subscription(Float32MultiArray,'/targets_visualization',self.visualization_callback,10)
        self.plot_timer = self.create_timer(0.01,self.plot_callback)
        self.img = None
        self.cnt = 10931
        self.array = None
        self.bridge = cv_bridge.CvBridge()
        self.get_logger().info("可视化启动成功")

    def img_callback(self,msg):
        try:
            self.img = self.bridge.compressed_imgmsg_to_cv2(msg)
            print("收到图像")
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")
    
    def visualization_callback(self,msg):
        self.array = msg.data
        print(f"收到数组数据: {self.array}")

    def plot_callback(self):
        if self.img is not None and self.array is not None:
            try:
                img = self.img.copy()
                # 保存图片
                if self.cnt % 10 == 0:
                    cv2.imwrite(f'/home/rdk/dataset/{self.cnt//10}.jpg', img)
                # 修正：直接访问一维数组元素
                # 根据发布端代码：[target_x, max_obs_x, fobbiden_scale]
                target_x = int(self.array[0])
                max_obs_x = self.array[1] if len(self.array) > 1 else None
                fobbiden_scale = self.array[2] if len(self.array) > 2 else None
                
                # 绘制目标点（红色圆圈）
                cv2.circle(img, (target_x, int(img.shape[0]/2)), 5, (0,0,255), 2)
                
                # 如果有障碍物信息，绘制禁区线（绿色线）
                if max_obs_x is not None and max_obs_x < 1000000000.0:  # 检查是否为有效值
                    max_obs_x = int(max_obs_x)
                    if fobbiden_scale is not None and fobbiden_scale < 1000000000.0:
                        fobbiden_scale = int(fobbiden_scale)
                        # 绘制禁区范围
                        left_bound = max(0, max_obs_x - fobbiden_scale)
                        right_bound = min(img.shape[1]-1, max_obs_x + fobbiden_scale)
                        cv2.line(img, (left_bound, int(img.shape[0]/2)), 
                                (right_bound, int(img.shape[0]/2)), 
                                color=(0,255,0), thickness=3)
                        
                        # 绘制障碍物中心点（蓝色）
                        cv2.circle(img, (max_obs_x, int(img.shape[0]/2)), 3, (255,0,0), 2)
                self.cnt+=1
                cv2.imshow("Visualization", img)
                # 修正：使用1毫秒等待，不阻塞程序
                cv2.waitKey(1)
                
            except Exception as e:
                self.get_logger().error(f"绘制失败: {e}")

if __name__ == "__main__":
    rclpy.init()
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()