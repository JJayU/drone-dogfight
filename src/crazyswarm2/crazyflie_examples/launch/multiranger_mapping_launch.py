import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    crazyflie = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('crazyflie'), 'launch'),
            '/launch.py']),
        launch_arguments={
            'backend': 'cflib',
            'gui': 'false',
            'teleop': 'false',
            'mocap': 'false',
            }.items())

    return LaunchDescription([
        crazyflie,
        Node(
            package='crazyflie',
            executable='vel_mux.py',
            name='vel_mux',
            output='screen',
            parameters=[{'hover_height': 0.3},
                        {'incoming_twist_topic': '/cmd_vel'},
                        {'robot_prefix': '/cf231'}]
        ),
        Node(
            parameters=[
                {'odom_frame': 'cf231/odom'},
                {'map_frame': 'map'},
                {'base_frame': 'cf231'},
                {'scan_topic': '/cf231/scan'},
                {'use_scan_matching': False},
                {'max_laser_range': 3.5},
                {'resolution': 0.1},
                {'minimum_travel_distance': 0.01},
                {'minimum_travel_heading': 0.001},
                {'map_update_interval': 0.1}
            ],
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen'),
    ])
