#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PointStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from tf_transformations import euler_from_quaternion
import math


class TFPublisherNode(Node):
    def __init__(self):
        super().__init__('tf_publisher_node')

        # Subskrypcje
        self.gps_sub = self.create_subscription(PointStamped, '/gps', self.gps_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)

        # TransformBroadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Dane pozycji i orientacji
        self.drone_pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.current_yaw = 0.0

    def gps_callback(self, msg: PointStamped):
        # Aktualizuj pozycję drona na podstawie GPS
        self.drone_pos['x'] = msg.point.x
        self.drone_pos['y'] = msg.point.y
        self.drone_pos['z'] = msg.point.z

        # Publikuj transformację
        self.publish_transform()

    def imu_callback(self, msg: Imu):
        # Aktualizuj orientację drona na podstawie IMU
        orientation = msg.orientation
        _, _, yaw = euler_from_quaternion([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ])
        self.current_yaw = yaw

    def publish_transform(self):
        # Przygotuj transformację
        transform = TransformStamped()

        # Ustaw nazwy układów współrzędnych
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "map"  # Globalny układ odniesienia
        transform.child_frame_id = "base_crazyflie"  # Lokalny układ drona

        # Ustaw pozycję
        transform.transform.translation.x = self.drone_pos['x']
        transform.transform.translation.y = self.drone_pos['y']
        transform.transform.translation.z = self.drone_pos['z']

        # Ustaw orientację (konwersja yaw na quaternion)
        quaternion = [0.0, 0.0, math.sin(self.current_yaw / 2), math.cos(self.current_yaw / 2)]
        transform.transform.rotation.x = quaternion[0]
        transform.transform.rotation.y = quaternion[1]
        transform.transform.rotation.z = quaternion[2]
        transform.transform.rotation.w = quaternion[3]

        # Opublikuj transformację
        self.tf_broadcaster.sendTransform(transform)

        t = TransformStamped()

        # Nagłówek
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_crazyflie'  # Układ odniesienia
        t.child_frame_id = 'laser_link'       # Układ lasera

        # Transformacja (pozycja i orientacja)
        t.transform.translation.x = 2.5
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.02
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.7071068
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 0.7071068

        # Publikacja transformacji
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
