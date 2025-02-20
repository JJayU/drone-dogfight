from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node

def generate_launch_description():

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['--display-config', '/home/ws/src/drone_tf_publisher/rviz/rviz.rviz']
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

    ld = LaunchDescription()
    
    ld.add_action(rviz_node)

    ld.add_action(TimerAction(
        period=2.0,
        actions=[tf_publisher_node, target_publisher_node]
    ))
    
    return ld