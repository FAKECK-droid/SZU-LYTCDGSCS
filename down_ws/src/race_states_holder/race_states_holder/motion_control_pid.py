from rclpy.node import Node
import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8,Float32MultiArray
import numpy as np
import threading
from race_states_holder.pid import SimplePID
from race_states_holder.utils import *
from ai_msgs.msg import PerceptionTargets
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class AkmControlNode(Node):
    def __init__(self):
        super().__init__('car_controller')
        self.target_velocity = 0.0
        #########################################################################
        #控制器
        #self.pid_vertical = SimplePID(0.0,0.0,0.0,dead_zone=0.01,integration_seprate=1.0,integration_bound=(-1.0,1.0),alpha=0.8,output_limits=(-1.2,1.2))
        self.pid_horical = SimplePID(0.009,0.0,0.0006,dead_zone=10,integration_bound=(-3.6,3.6),alpha=0.1,output_limits=(-3.0,3.0))
        #########################################################################
        #订阅话题
        self.lock = threading.Lock()                    #互斥锁
        self.odom_flag = False                          #第一条里程计消息标记位
        self.odom_sub = self.create_subscription(Odometry,'/odom_combined', self.odom_callback,10)   
        # self.point_sub = self.create_subscription(Pose2D,'/target_point', self.point_callback,10) #目标点订阅（用于调试）
        self.imu_sub = self.create_subscription(Imu,'/imu/data', self.imu_callback,10)
        # self.target_velocity_sub = self.create_subscription(Float64,'/target_velocity', self.target_velocity_callback,10) #目标速度订阅（用于调试）
        self.targets_sub = self.create_subscription(PerceptionTargets,'/hobot_dnn_detection',self.targets_callback,10) #目标检测结果话题订阅
        self.stop_sub = self.create_subscription(Int8,'/stop',self.stop_callback,10)     #急停话题订阅
        self.state_sub =self.create_subscription(Int8,'/state',self.state_callback,10)   #状态话题订阅

        #########################################################################
        #发布话题
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        self.cmd_pub = self.create_publisher(Twist,'/cmd_vel',qos_profile=qos_profile)
        self.targets_visualization_pub = self.create_publisher(Float32MultiArray,'/targets_visualization',10)   #上位机可视化信息订阅
        #########################################################################
        #控制循环
        self.control_timer = self.create_timer(0.05, self.control_callback)
        ##########################################################################
        #小车基础设置
        self.camera_K = np.array([[301.9237, 0.0, 295.2176],[0.0,302.0786,202.9267],[0.0,0.0,1.0]]) #相机内参矩阵

        self.current_states = np.float64([0,0,0,0,0]) #小车状态[x,y,theta,v,w]
        self.target_point = np.float64([4.65,0.0,0.0]) 
        self.target_point_task_1  = np.float64([4.65,0.0,0.0]) #任务1目标位姿  
                                                                              
        self.target_point_task_3 = np.float64([3.5,0.0,0.0]) #任务3目标位姿
        self.target_vel = 1.2 #目标速度

        #惯导相关变量
        self.fobbiden_gain = 1.35 #禁区增益
        self.fobbiden_scale_max = 260
        self.last_hesitation_err_max = 30
        
        self.height_max = None #最大障碍物高度
        self.max_obs = None
        self.fobbiden_ranges = []
        
        self.qrcode_x = None #二维码x坐标（画面坐标系）
        self.end_x = None #停车位x坐标（画面坐标系）

        self.last_hesitation_err  = None  #犹豫误差
        self.last_obs_dir = None #上一次避障方向

        self.stop_flag = 1 #急停标志
        self.state = 0 #状态
        self.get_logger().info("控制节点已启动")
        ##########################################################################
    
    ############################################################################
    #小车状态更新
    def odom_callback(self,msg):
        with self.lock:
            if not self.odom_flag:
                self.odom_flag = True
                self.get_logger().info("收到第一条里程计消息")
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            theta = np.arctan2(2*orientation.z*orientation.w,1-2*orientation.z**2)
            self.current_states[2] = theta

            self.current_states[0] = position.x
            self.current_states[1] = position.y
            self.current_states[3] = msg.twist.twist.linear.x
                
            # print(self.cur_states[4])
            # print(theta)
            
    def imu_callback(self,msg):
        with self.lock:
            self.current_states[4] = msg.angular_velocity.z

    #目标点更新
    def point_callback(self,msg):
        self.target_point[0] = msg.x
        self.target_point[1] = msg.y
        self.target_point[2] = msg.theta
        print("目标点",self.target_point)

    ###########################################################################
    #目标检测回调函数
    def targets_callback(self,msg):
        targets = msg.targets
        self.fobbiden_ranges = []
        
        self.line_most_x = None
        self.qrcode_x = None
        self.end_x = None
        
        for target in targets:
            if target.type == "roadblock":
                if target.rois[0].confidence > 0.5:
                    #障碍物检测框信息
                    width = target.rois[0].rect.width
                    height = target.rois[0].rect.height
                    x_offset = target.rois[0].rect.x_offset + width / 2
                    y_offset = target.rois[0].rect.x_offset + height / 2
                    
        
                    fobbiden_scale = self.fobbiden_gain * height
                    if fobbiden_scale > self.fobbiden_scale_max:
                        fobbiden_scale = self.fobbiden_scale_max

                    self.fobbiden_ranges.append((x_offset - fobbiden_scale, 
                                                 x_offset + fobbiden_scale, 
                                                 height, 
                                                 width, 
                                                 x_offset))

                    
            elif target.type == "qrcode":
                # print("检测到二维码")
                if self.state == 1:
                    if calculate_dis(self.current_states[:2],self.target_point_task_1[:2]) < 1.5: 
                        
                        self.set_params(horical_pid=(0.0083,0.0,0.0),target_vel=0.5,
                                        fobbiden_gain=1.45,fobbiden_scale_max=280)
    
                        if target.rois[0].confidence > 0.4:
                            self.qrcode_x = target.rois[0].rect.x_offset 
                            print("更新目标点为二维码位置")
                            
            elif target.type == 'end':
                # print("检测到停车位")
                if self.state == 3:
                    if target.rois[0].rect.width * target.rois[0].rect.height > 2000:
                        
                        self.set_params(horical_pid=(0.0083,0.0,0.0),target_vel=0.5,
                                        fobbiden_gain=1.45,fobbiden_scale_max=280)
                        
                        # print("减速到0.5")
                        if target.rois[0].confidence > 0.5:
                            self.end_x = target.rois[0].rect.x_offset + target.rois[0].rect.width / 2
                            print("更新目标点为停车位位置")

    #手动停止
    def stop_callback(self,msg):
        if msg.data == 1:
            self.stop_flag = 1
            # print("手动停止")
        else:
            self.stop_flag = 0
            # print("启动")

    #状态回调
    def state_callback(self,msg):
        if msg.data == 0:
            self.qrcode_x = None
            self.end_x = None
            self.qr_flag = False

        elif msg.data == 1:
            
            self.set_params(horical_pid=(0.0083,0.0,0.0),target_vel=1.2,
                            fobbiden_gain=1.35,fobbiden_scale_max=250)
            
            self.fobbiden_ranges = []
            self.last_hesitation_err = None
            self.last_obs_dir = None
            
            self.target_point = self.target_point_task_1 
            print("自动驾驶状态")
            
        elif msg.data == 2:
            self.load_zero_control()
            print("识别到二维码或手动切换遥控模式")
            
        elif msg.data == 3:
            
            self.set_params(horical_pid=(0.0083,0.0,0.0),target_vel=1.2,
                            fobbiden_gain=1.35,fobbiden_scale_max=250)
            
            self.fobbiden_ranges = []
            self.last_hesitation_err = None
            self.last_obs_dir = None
            
            self.target_point = self.target_point_task_3
            print("返回停车位状态")

        self.state = msg.data
    ############################################################################
    # def target_velocity_callback(self,msg):
    #     with self.lock:
    #         self.target_velocity = msg.data     
        # print("target_velocity",self.target_velocity)
    ############################################################################
    # 控制回调函数
    def control_callback(self):
        #######################################################
        #pid惯导
        angle_target_x = 320
        self.height_max = 0.0
        self.max_obs = None
        if self.state == 1 and self.qrcode_x is not None:
            self.target_point = calculate_target_point(self.camera_K,self.qrcode_x,
                                                       self.current_states,dis=calculate_dis(self.current_states[:2],
                                                       self.target_point_task_1[:2])+1.0)   #假设二维码位于1.0m远的位置
            
        elif self.state == 3 and self.end_x is not None:
            self.target_point = calculate_target_point(self.camera_K,self.end_x,
                                                       self.current_states,dis=calculate_dis(self.current_states[:2],
                                                       self.target_point_task_3[:2])+1.0)    #假设停车位位于1.0m远的位置
            

        target_vector = self.target_point[:2] - self.current_states[:2]
        
        angle_err = np.arctan2(target_vector[1], target_vector[0]) - self.current_states[2]     #角度形式的pid误差
        if angle_err > np.pi:
            angle_err -= 2 * np.pi
        elif angle_err < -np.pi:
            angle_err += 2 * np.pi 

        angle_target_x = calculate_imu_target(angle_err, self.camera_K, (640,480))[0]           # 映射到画面上
        
        for i in self.fobbiden_ranges:
            if (i[2] > self.height_max) and (i[0] < angle_target_x < i[1]) and (i[2] / i[3] < 1.8):
                self.max_obs = i
                self.height_max = i[2]
                
        #禁区检测
        target_x = angle_target_x
        if self.max_obs is not None:
            
            #障碍物在目标方向右边
            if self.max_obs[0] < angle_target_x < self.max_obs[4]:
                target_x = self.max_obs[0]
                # print('left')
                hesitation_err = angle_target_x - self.max_obs[4]
                #犹豫检测
                if self.last_hesitation_err is not None and self.last_obs_dir is not None:
                    
                    if np.fabs(hesitation_err - self.last_hesitation_err) < self.last_hesitation_err_max: 
                        if self.last_obs_dir == 'right':
                            target_x = self.max_obs[1]
                            # print("犹豫检测，保持右转避障,target_x:",target_x)
                    else:
                        self.last_obs_dir ='left'
                        # print("左转避障，target_x:",target_x)
                else:
                    self.last_obs_dir = 'left'
                    # print("左转避障，target_x:",target_x)
                self.last_hesitation_err = hesitation_err  

            #障碍物在目标方向左边或中间
            elif self.max_obs[4] <= angle_target_x < self.max_obs[1]:
                target_x = self.max_obs[1]
                # print('right')
                hesitation_err = angle_target_x - self.max_obs[4]
                #犹豫检测
                if self.last_hesitation_err is not None and self.last_obs_dir is not None:
            
                    if np.fabs(hesitation_err - self.last_hesitation_err) < self.last_hesitation_err_max:
                        if self.last_obs_dir == 'left':
                            target_x = self.max_obs[0]

                    else:
                        self.last_obs_dir ='right'
                        # print("右转避障，target_x:",target_x)
                else:
                    self.last_obs_dir = 'right'
                    # print("右转避障，target_x:",target_x)
                self.last_hesitation_err = hesitation_err
                      
            
        # print("target_x=",target_x)
        #横向误差（像素坐标值）
        horical_err = -(target_x - 320)
        
        #发送给上位机的可视化信息
        visualization_msg = Float32MultiArray()
        if self.max_obs is not None:
            visualization_msg.data = [float(target_x),float(self.max_obs[4]),float((self.max_obs[1] - self.max_obs[0]) / 2)]
        else:
            visualization_msg.data = [float(target_x),1000000000.0,0.0]
        self.targets_visualization_pub.publish(visualization_msg)

        # print(visualization_msg.data)
        
        w_cmd = self.pid_horical.compute(horical_err)
        # print("w_cmd:",w_cmd)
        # print(self.goal_check())
        # print("state=",self.state)
        # print("stop=",self.stop_flag)


        #加载控制量
        if self.stop_flag == 0 and (self.state == 1 or self.state == 3):
            if not self.goal_check():
                self.load_control(self.target_vel,w_cmd)
            else:
                self.load_control(self.target_vel,w_cmd)# self.load_zero_control()
                self.get_logger().info("接近目标点")
        
        elif (self.stop_flag == 1 and self.state != 2) or self.state == 0:

            self.load_zero_control()

        elif self.state == 2:
            # print("当前状态",self.state)
            pass
        
            
    def load_control(self,v,w):
        msg = Twist()
        msg.linear.x = v
        msg.linear.y = 0.0
        msg.linear.z = 0.0

        msg.angular.z = w
        msg.angular.y = 0.0
        msg.angular.x = 0.0
        self.cmd_pub.publish(msg)
    
    def load_zero_control(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0

        msg.angular.z = 0.0
        msg.angular.y = 0.0
        msg.angular.x = 0.0
        self.cmd_pub.publish(msg)
    
    """
    检查小车是否接近目标点，接近就返回True，然后降速
    """
    def goal_check(self):

        s = np.dot((self.target_point[:2] - self.current_states[:2]),
                                np.array([np.cos(self.current_states[2]),np.sin(self.current_states[2])]))

        dis = 0.0

        if self.state == 1:
            dis = calculate_dis(self.current_states,self.target_point_task_1)
            
            if dis < 1.0:      
              self.set_params(horical_pid=(0.0083,0.0,0.0),target_vel=0.3)
              
        elif self.state == 3:
            dis = calculate_dis(self.current_states,self.target_point_task_3)
        
        return dis < 0.3


    def set_params(self,**kwargs):
        p,i,d = kwargs.get('horical_pid',self.pid_horical.get_pid())
        self.pid_horical.reset(kp=p,ki=i,kd=d)
        self.fobbiden_gain = kwargs.get('fobbiden_gain',self.fobbiden_gain)
        self.fobbiden_scale_max = kwargs.get('fobbiden_scale_max',self.fobbiden_scale_max)
        self.target_vel = kwargs.get('target_vel',self.fobbiden_gain)


def main():
    rclpy.init()
    node = AkmControlNode()
    rclpy.spin(node)
    node.__del__()
    rclpy.shutdown()

if __name__ == "__main__":
    main()