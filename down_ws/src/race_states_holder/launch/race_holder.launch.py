import os
from pathlib import Path
import launch
from launch.actions import SetEnvironmentVariable
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, SetEnvironmentVariable, ExecuteProcess)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import PushRosNamespace
import launch_ros.actions
from launch.conditions import UnlessCondition

def generate_launch_description():
    # Get the launch directory
    bringup_dir = get_package_share_directory('origincar_base')
    launch_dir = os.path.join(bringup_dir, 'launch')
    camera_dir = get_package_share_directory('origincar_bringup')
    camera_launch_dir = os.path.join(camera_dir, 'launch')

    rosbridge = ExecuteProcess(
            cmd=['ros2', 'launch', 'rosbridge_server', 'rosbridge_websocket_launch.xml'],
            output='screen'
        )

    origincar_base_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_dir, 'origincar_bringup.launch.py')),
    )

    camera_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(camera_launch_dir, 'usb_websocket_display.launch.py'))
    )

    qwen_client_node = launch_ros.actions.Node(
        package='qwen_vl_client',
        executable='qwen_vl_client'
    )
                                
    motion_control_node = launch_ros.actions.Node(
            package='race_states_holder', 
            executable='motion_control', 
            name='car_controller',
    )

    state_holder_node = launch_ros.actions.Node(
            package='race_states_holder', 
            executable='states_holder',
    )

    qrcode_node = launch_ros.actions.Node(
            package='qrcode_detection_two', 
            executable='erweima',
    )
    
    qrcode_node_opencv = launch_ros.actions.Node(
            package='qrcode_detection_two', 
            executable='erweima_opencv',
    )

    ld = LaunchDescription()

    ld.add_action(rosbridge)
    ld.add_action(origincar_base_launch)
    ld.add_action(camera_launch)
#     ld.add_action(motion_control_node)
#     ld.add_action(state_holder_node)
    ld.add_action(qrcode_node)
    ld.add_action(qrcode_node_opencv)
    ld.add_action(qwen_client_node)

    return ld