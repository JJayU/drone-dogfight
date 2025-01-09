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

    cf_examples_dir = get_package_share_directory('crazyflie_examples')
    bringup_dir = get_package_share_directory('nav2_bringup')
    bringup_launch_dir = os.path.join(bringup_dir, 'launch')
    map_name = 'map'

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
                {'scan_topic': 'cf231/scan'},
                {'use_scan_matching': False},
                {'max_laser_range': 3.5},
                {'resolution': 0.1},
                {'minimum_travel_distance': 0.01},
                {'minimum_travel_heading': 0.001},
                {'map_update_interval': 0.1},
                {'mode': 'localization'},
                {'map_file_name': cf_examples_dir + '/data/' + map_name},
                {'map_start_pose': [0.0, 0.0, 0.0]}],
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bringup_launch_dir, 'bringup_launch.py')),
            launch_arguments={
                'slam': 'False',
                'use_sim_time': 'False',
                'map': cf_examples_dir + '/data/' + map_name + '.yaml',
                'params_file': os.path.join(cf_examples_dir, 'config/nav2_params.yaml'),
                'autostart': 'True',
                'use_composition': 'True',
                'transform_publish_period': '0.02'
            }.items()
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bringup_launch_dir, 'rviz_launch.py')),
            launch_arguments={
                'rviz_config': os.path.join(
                    bringup_dir, 'rviz', 'nav2_default_view.rviz')}.items())
    ])
