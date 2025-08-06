from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='image_resize_node',
            executable='image_resize_node',
            name='image_resize_node',
            output='screen',
            parameters=[
                {'use_sim_time': False}
            ]
        ),
    ])
