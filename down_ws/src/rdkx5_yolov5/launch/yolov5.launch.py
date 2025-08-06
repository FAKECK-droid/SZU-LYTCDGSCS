from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        # 参数声明
        DeclareLaunchArgument(
            'model_path',
            default_value='/root/new_ws/src/rdkx5_yolov5/models/detect.bin',
            description='Path to YOLOv5 model file'),
            
        DeclareLaunchArgument(
            'classes_num',
            default_value='4',
            description='Number of detection classes'),
            
        DeclareLaunchArgument(
            'nms_threshold',
            default_value='0.45',
            description='NMS threshold'),
            
        DeclareLaunchArgument(
            'score_threshold',
            default_value='0.25',
            description='Score threshold'),
            
        DeclareLaunchArgument(
            'nms_top_k',
            default_value='300',
            description='Maximum number of boxes to keep after NMS'),
            
        DeclareLaunchArgument(
            'enable_draw',
            default_value='true',
            description='Enable result visualization'),
        
        # 节点启动
        Node(
            package='rdkx5_yolov5',
            executable='yolov5_node',
            name='yolov5_detector',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'classes_num': LaunchConfiguration('classes_num'),
                'nms_threshold': LaunchConfiguration('nms_threshold'),
                'score_threshold': LaunchConfiguration('score_threshold'),
                'nms_top_k': LaunchConfiguration('nms_top_k'),
                'enable_draw': LaunchConfiguration('enable_draw'),
            }]
        ),
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'rdkx5_yolov5', 'yolov5_web_server',
                'root/new_ws/src/rdkx5_yolov5/models/detect.bin', '4', '0'
            ],
            output='screen'
        ) 
    ])