from launch import LaunchDescription
from launch.actions import TimerAction, DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition

#
# Visualization launch file, used to launch Rviz2, tf_publisher and target_publisher (if not cancelled by setting `publish_target` to False)
#

def generate_launch_description():
    
    publish_target = LaunchConfiguration('publish_target')
    
    publish_target_arg = DeclareLaunchArgument(
        'publish_target', default_value='True', description='Whether to launch target_publisher_node'
    )

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
        name='target_publisher',
        condition=IfCondition(
            PythonExpression([
                publish_target
            ]))
    )

    ld = LaunchDescription()
    
    ld.add_action(rviz_node)
    
    ld.add_action(publish_target_arg)

    ld.add_action(TimerAction(
        period=2.0,
        actions=[tf_publisher_node, target_publisher_node]
    ))
    
    return ld