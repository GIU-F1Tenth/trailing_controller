from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('trailing_controller'),
        'config',
        'trailing_params.yaml'
    )
    
    trailing_controller_node = Node(
        package='trailing_controller',
        executable='trailing_controller_node',
        name='trailing_controller_node',
        parameters=[config],
        output='screen'
    )
    
    return LaunchDescription([
        trailing_controller_node
    ])
