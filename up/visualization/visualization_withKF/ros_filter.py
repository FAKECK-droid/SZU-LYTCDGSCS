import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
import cv_bridge
from Tracker import Tracker
from std_msgs.msg import Float32MultiArray

class KalmanFilterNode(Node):
    def __init__(self):
        super().__init__("KalmanFilter")
        self.tracker = Tracker()

        self.last_time = None
        self.tracks = None
        self.targets = []
        self.target_sub = self.create_subscription(PerceptionTargets,"/hobot_dnn_detection",self.target_callback,10)
        
        self.img = None
        self.img_sub = self.create_subscription(CompressedImage,"/image",self.img_callback,10)
        self.array = None
        self.visualization_sub = self.create_subscription(Float32MultiArray,'/targets_visualization',self.visualization_callback,10)
        self.bridge = cv_bridge.CvBridge()
        self.plot_timer = self.create_timer(0.01,self.plot_callback)
        self.get_logger().info("滤波器节点启动成功")
    
    def img_callback(self,msg):
        try:
            self.img = self.bridge.compressed_imgmsg_to_cv2(msg)
            # print("收到图像")
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")
    
    def visualization_callback(self,msg):
        self.array = msg.data
        # print(f"收到数组数据: {self.array}")
    
    def target_callback(self,msg):
        sec,nanosec = self.get_clock().now().seconds_nanoseconds()
        targets_msg = msg.targets
        self.targets = []
        for target in targets_msg:
            if target.type == "roadblock":
                target = np.array([target.rois[0].rect.x_offset,
                                         target.rois[0].rect.y_offset,
                                         target.rois[0].rect.width,
                                         target.rois[0].rect.height])
                self.targets.append(target)
        # print("targets:\n",targets)
        if self.last_time is None:
            self.tracks = np.array(self.tracker.track(self.targets))
            # print("self.tracks:\n",self.tracks)
            
        else:
            self.tracks = np.array(self.tracker.track(self.targets,dt=sec+nanosec*1e-9 - self.last_time))
            # print("self.tracks:\n",self.tracks)

        self.last_time = sec + nanosec*1e-9

    
    def plot_callback(self):
        if self.img is not None and self.tracks is not None:
            img = self.img.copy()
            if len(self.tracks) != 0:
                self.pt1 = self.tracks[:,:2]
                self.pt2 = self.pt1 + self.tracks[:,2:4]
                # print(self.pt1)
                # print(self.pt2)
                for pt1, pt2 in zip(self.pt1, self.pt2):
                    pt1_tuple = tuple(map(int, pt1))
                    pt2_tuple = tuple(map(int, pt2))
                    cv2.rectangle(img, pt1_tuple, pt2_tuple, (255,255,0), 2)
            
            if len(self.targets) != 0:
                self.pt1 = np.array(self.targets)[:,:2]
                self.pt2 = self.pt1 + np.array(self.targets)[:,2:4]
                # print(self.pt1)
                # print(self.pt2)
                for pt1, pt2 in zip(self.pt1, self.pt2):
                    pt1_tuple = tuple(map(int, pt1))
                    pt2_tuple = tuple(map(int, pt2))
                    cv2.rectangle(img, pt1_tuple, pt2_tuple, (255,0,255), 2)
                    
            # 根据发布端代码：[target_x, max_obs_x, fobbiden_scale]
            if self.array is not None:
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
                        
            cv2.imshow("Visualization", img)
            cv2.waitKey(1)

if __name__ == "__main__":
    rclpy.init()
    node = KalmanFilterNode()
    rclpy.spin(node)
    rclpy.shutdown()
