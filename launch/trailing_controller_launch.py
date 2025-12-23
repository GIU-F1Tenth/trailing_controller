from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('trailing_controller')
    config_file = os.path.join(pkg_dir, 'config', 'trailing_controller_params.yaml')
    
    return LaunchDescription([
        Node(
            package='trailing_controller',
            executable='trailing_controller_node',
            name='trailing_controller_node',
            output='screen',
            parameters=[config_file],
        ),
    ])
