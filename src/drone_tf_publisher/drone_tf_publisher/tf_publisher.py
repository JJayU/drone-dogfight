#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PointStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from tf_transformations import euler_from_quaternion
import math

#
# Script used for publishing a tf transform based on actual position and orientation of the drone
#

class TFPublisherNode(Node):
    def __init__(self):
        super().__init__('tf_publisher_node')

        self.gps_sub = self.create_subscription(PointStamped, '/gps', self.gps_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.drone_pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.current_yaw = 0.0
        
        self.quat = [0.,0.,0.,1.]

    def gps_callback(self, msg: PointStamped):
        # Update drone position
        self.drone_pos['x'] = msg.point.x
        self.drone_pos['y'] = msg.point.y
        self.drone_pos['z'] = msg.point.z

        self.publish_transform()

    def imu_callback(self, msg: Imu):
        # Update drone orientation
        orientation = msg.orientation
        self.quat = [orientation.x, orientation.y, orientation.z, orientation.w]

    def publish_transform(self):
        
        # Drone tf
        transform = TransformStamped()

        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "map"               # Global coordinate system
        transform.child_frame_id = "base_crazyflie"     # Local coordinate system

        transform.transform.translation.x = self.drone_pos['x']
        transform.transform.translation.y = self.drone_pos['y']
        transform.transform.translation.z = self.drone_pos['z']

        transform.transform.rotation.x = self.quat[0]
        transform.transform.rotation.y = self.quat[1]
        transform.transform.rotation.z = self.quat[2]
        transform.transform.rotation.w = self.quat[3]

        self.tf_broadcaster.sendTransform(transform)

        t = TransformStamped()

        # Laserbeam tf
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_crazyflie'    # Drone coord. system
        t.child_frame_id = 'laser_link'         # Laser coord. system

        t.transform.translation.x = 2.5
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.02        # Height of laserbeam in reference to drone base
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.7071068
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 0.7071068

        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = TFPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
