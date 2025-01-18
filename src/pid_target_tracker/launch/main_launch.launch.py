import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

def generate_launch_description():

    webots_crazyflie = ExecuteProcess(
        cmd=['ros2', 'launch', 'webots_ros2_crazyflie', 'robot_launch.py'],
        output='screen'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['--display-config', 'src/pid_target_tracker/rviz/rviz.rviz']
    )
    
    pid_tracker_node = Node(
        package='pid_target_tracker',
        executable='pid_tracker',
        name='pid_tracker'
    )

    tf_publisher_node = Node(
        package='drone_tf_publisher',
        executable='tf_publisher',
        name='tf_publisher'
    )

    target_publisher_node = Node(
        package='drone_tf_publisher',
        executable='target_publisher',
        name='target_publisher'
    )

    
    # Tworzymy opis launch
    ld = LaunchDescription()
    
    # Dodajemy node'y z małymi opóźnieniami, aby zapewnić poprawne uruchomienie
    ld.add_action(webots_crazyflie)  # Najpierw Webots
    
    # Dodajemy RViz2 z opóźnieniem 2 sekund
    ld.add_action(TimerAction(
        period=2.0,
        actions=[rviz_node]
    ))

    ld.add_action(TimerAction(
        period=5.0,
        actions=[pid_tracker_node, tf_publisher_node, target_publisher_node]
    ))
    
    return ld
