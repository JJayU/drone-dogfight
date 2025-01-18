#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PointStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from tf_transformations import euler_from_quaternion
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import math


class TargetPublisherNode(Node):
    def __init__(self):
        super().__init__('target_publisher_node')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        # Parametry trajektorii
        self.t = 0.0
        self.square_size = 0.8  # Rozmiar ruchu w metrach
        self.z_base = 1.0      # Wysokość podstawowa
        self.speed = 0.5       # Prędkość ruchu (mniejsza = wolniej)
        self.segment_time = 4.0 # Czas na jeden segment ruchu

        self.target_pub = self.create_publisher(PointStamped, '/target_point', qos)
        self.create_timer(0.02, self.loop)

    def calculate_position(self):
        # Całkowity czas na pełny cykl
        total_time = self.segment_time * 4  # 4 segmenty ruchu
        current_t = self.t % total_time
        segment = int(current_t / self.segment_time)
        segment_t = (current_t % self.segment_time) / self.segment_time
        
        # Płynne przejście w ramach segmentu
        smooth_t = (1 - math.cos(segment_t * math.pi)) / 2
        
        x = 0.0
        y = 0.0
        z = self.z_base
        
        # Prosty ruch: góra -> prawo -> dół -> lewo
        if segment == 0:    # Ruch w górę
            x = 0.0
            y = 1.0
            z = self.z_base + self.square_size * smooth_t
        elif segment == 1:  # Ruch w prawo
            x = self.square_size * smooth_t
            y = 1.0
            z = self.z_base + self.square_size
        elif segment == 2:  # Ruch w dół
            x = self.square_size
            y = 1.0
            z = self.z_base + self.square_size * (1 - smooth_t)
        else:              # Ruch w lewo
            x = self.square_size * (1 - smooth_t)
            y = 1.0
            z = self.z_base
            
        return x, y, z


    def loop(self):

        current_time = self.get_clock().now()

        x, y, z = self.calculate_position()

        point_msg = PointStamped()
        point_msg.header.stamp = current_time.to_msg()
        point_msg.header.frame_id = "map"
        point_msg.point.x = x
        point_msg.point.y = y
        point_msg.point.z = z
        
        self.target_pub.publish(point_msg)

        self.t += self.speed * 0.02


def main():
    rclpy.init()
    node = TargetPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
