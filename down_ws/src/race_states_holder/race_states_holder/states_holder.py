import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8,String,Int32,Header
from geometry_msgs.msg import PoseWithCovarianceStamped

class StateNode(Node):
    def __init__(self):
        super().__init__("state_holder")
        self.state = 0
        self.dict = {0:"停车",1:"执行任务1",2:"切换遥操作执行任务2",3:"执行任务3返回停车位"}

        self.qrcode_msg = None
        
        self.block_qrcode_msg = String()
        self.block_qrcode_msg.data = " "

        self.qrcode_pub = self.create_publisher(String,"/qrcode_info",10)
        self.state_pub = self.create_publisher(Int8,"/state",10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped,"/set_pose",10)
        self.return_pub = self.create_publisher(Int32,"/sign_foxglove_1",10)

        #订阅话题
        self.qrcode_sub = self.create_subscription(String,"/qrcode_info",self.qrcode_callback,10)
        self.return_sub = self.create_subscription(Int32,"/sign4return",self.sign4return_callback,10)
        
        print(self.dict[self.state])


    def qrcode_callback(self,msg):
        self.qrcode_msg = msg.data
        if self.qrcode_msg is not None:
            if self.state == 1:
                self.get_logger().info(f"识别到二维码：{self.qrcode_msg}")
                
                if self.qrcode_msg.isdigit():
                    
                    if int(self.qrcode_msg) % 2 == 0:
                        sign_pub = Int32()
                        sign_pub.data = 4
                        self.return_pub.publish(sign_pub)
                    else:
                        sign_pub = Int32()
                        sign_pub.data = 3
                        self.return_pub.publish(sign_pub)
                
                else:
                    self.get_logger().info("二维码内容不是数字，无法处理")

                self.state = 2
                msg_state_pub = Int8()
                msg_state_pub.data = self.state
                self.state_pub.publish(msg_state_pub)
                print(self.dict[self.state])

    def sign4return_callback(self,msg):
        # 停车
        if msg.data == 0:
            self.reset_odom()
            
            self.state = 0

            self.state_pub.publish(Int8(data=self.state))

            self.qrcode_pub.publish(self.block_qrcode_msg)
            
            self.return_pub.publish(Int32(data=0))

            self.get_logger().info(self.dict[self.state])

        # 执行任务1
        elif msg.data == 1:
            self.reset_odom()

            self.state = 1

            self.state_pub.publish(Int8(data=self.state))

            self.return_pub.publish(Int32(data=1))
            
            self.get_logger().info(self.dict[self.state])

        # 切换遥操作执行任务2
        elif msg.data == 5:
            # self.reset_odom()

            self.state = 2
            
            self.state_pub.publish(Int8(data=self.state))

            self.return_pub.publish(Int32(data=5))
            
            self.get_logger().info(self.dict[self.state])

        # 执行任务3返回停车位
        elif msg.data == 6:
            self.reset_odom()

            self.state = 3
            
            self.state_pub.publish(Int8(data=self.state))

            self.return_pub.publish(Int32(data=6))
            
            self.get_logger().info(self.dict[self.state])

        # 重置里程计
        elif msg.data == -1:
            self.return_pub.publish(Int32(data=-1))
            self.reset_odom()
    
    def reset_odom(self):
        """重置里程计，发布零值里程计消息"""
        zero_odom = PoseWithCovarianceStamped()
        
        # 设置消息头
        zero_odom.header = Header()
        zero_odom.header.stamp = self.get_clock().now().to_msg()
        zero_odom.header.frame_id = "odom_combined"
    
        # 初始化位置为零
        zero_odom.pose.pose.position.x = 0.0
        zero_odom.pose.pose.position.y = 0.0
        zero_odom.pose.pose.position.z = 0.0
        
        # 初始化姿态为零（四元数的w=1表示无旋转）
        zero_odom.pose.pose.orientation.x = 0.0
        zero_odom.pose.pose.orientation.y = 0.0
        zero_odom.pose.pose.orientation.z = 0.0
        zero_odom.pose.pose.orientation.w = 1.0
        
        # 发布零值里程计消息
        self.pose_pub.publish(zero_odom)
        print("里程计已重置")
    


def main(args=None):
    rclpy.init(args=args)
    node=StateNode()
    rclpy.spin(node) 
    node.destroy_node()
    rclpy.shutdown() 

if __name__ == "__main__":
    main()