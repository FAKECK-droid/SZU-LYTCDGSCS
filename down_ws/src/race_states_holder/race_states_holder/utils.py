import numpy as np
import math
from scipy.optimize import minimize_scalar

############################
def rotation_trans(vector, angle):
    """
    将二维向量旋转theta角度（弧度制）

    参数:
    vector: 二维向量 [x, y]
    angle: 旋转角度（弧度制），正值为逆时针旋转

    返回:
    rotated_vector: 旋转后的向量 [x', y']
    """
    x, y = vector
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    x_new = x * cos_theta - y * sin_theta
    y_new = x * sin_theta + y * cos_theta

    return np.array([x_new, y_new])
##################################################
def calculate_dis(pt1,pt2):
    dis = np.sqrt(np.sum(np.square(pt1[:2] - pt2[:2])))
    return dis

def calculate_theta_dif(theta1,theta2):
    dif = np.arctan2(np.sin(theta2-theta1),np.cos(theta2-theta1))
    return dif

def calculate_vector(point_start,point_end):
    vector = point_end - point_start
    return vector

def normlize_vector(vector):
    return vector / np.linalg.norm(vector)
###################################################
def calculate_imu_target(angle_theta, camera_matrix, image_size):
    """
    使用完整的相机参数矩阵进行计算，解决tan函数的值域溢出和周期性问题
    对于后方目标，映射到最近的边缘像素坐标

    参数:
    angle_theta: 水平夹角 (弧度)
    camera_matrix: 3x3 相机内参矩阵 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    image_size: (width, height) 图像尺寸

    返回:
    pixel_x: 横坐标
    is_valid: 是否在正常视野范围内（True表示前方视野，False表示边缘映射）
    """

    # 提取相机参数
    fx = camera_matrix[0, 0]  # x方向焦距
    cx = camera_matrix[0, 2]  # 主点x坐标

    # 解决周期性问题：将角度规范化到 [-π, π] 范围
    angle_normalized =  angle_theta#math.atan2(math.sin(angle_theta), math.cos(angle_theta))

    # 图像边界
    left_edge = 0
    right_edge = image_size[0]

    # 检查目标是否在相机前方视野范围内（-90° 到 +90°）
    if abs(angle_normalized) >= math.pi / 2:
        # 目标在相机后方，需要映射到边缘
        if angle_normalized > 0:
            # 后方左侧 (90° 到 180°)，映射到左边缘
            pixel_x = left_edge
        else:
            # 后方右侧 (-90° 到 -180°)，映射到右边缘  
            pixel_x = right_edge
        
        return pixel_x, False

    # 前方目标处理
    # 检查是否接近tan函数的奇点（±90°），避免数值溢出
    angle_limit = math.pi / 2 - 0.001  # 约89.94°，很接近边界但避免溢出

    if abs(angle_normalized) >= angle_limit:
        # 角度非常接近±90°，直接映射到边缘
        if angle_normalized > 0:
            # 接近+90°（左侧边界）
            pixel_x = left_edge
        else:
            # 接近-90°（右侧边界）
            pixel_x = right_edge
        
        is_valid = 0 <= pixel_x <= image_size[0]
        return pixel_x, is_valid

    # 安全计算像素坐标
    pixel_x = cx - fx * math.tan(angle_normalized)

    # 检查是否在图像范围内
    is_valid = 0 <= pixel_x <= image_size[0]

    return pixel_x, is_valid

def calculate_target_point(camera_matrix,pixel_x,currrent_states,dis=0.5):
    target_point = np.array([0.0,0.0,0.0])
    # 提取相机参数
    fx = camera_matrix[0, 0]  # x方向焦距
    cx = camera_matrix[0, 2]  # 主点x坐标

    angle2front = math.atan((pixel_x - cx)/(-fx))
    angle = angle2front + currrent_states[2]

    target_point[:2] = currrent_states[:2] + dis*np.array([np.cos(angle),np.sin(angle)])

    return target_point







    