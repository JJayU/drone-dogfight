#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import math

#
# Script used for publishing a virtual target point for aiming
#

class TargetPublisherNode(Node):
    def __init__(self):
        super().__init__('target_publisher_node')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        # Trajectory params
        self.t = 0.0
        self.square_size = 0.8  
        self.z_base = 1.0      
        self.speed = 0.5       
        self.segment_time = 4.0 

        self.target_pub = self.create_publisher(PointStamped, '/target_point', qos)
        self.create_timer(0.02, self.loop)
        
        self.dynamic_target = False

    def calculate_position(self):
        
        if self.dynamic_target:
            # Dynamic target (square movement)
            
            total_time = self.segment_time * 4 
            current_t = self.t % total_time
            segment = int(current_t / self.segment_time)
            segment_t = (current_t % self.segment_time) / self.segment_time
        
            smooth_t = (1 - math.cos(segment_t * math.pi)) / 2
            
            x = 0.0
            y = 0.0
            z = self.z_base
            
            # Movement on each side of square
            if segment == 0:   
                x = 0.0
                y = 1.0
                z = self.z_base + self.square_size * smooth_t
            elif segment == 1:  
                x = self.square_size * smooth_t
                y = 1.0
                z = self.z_base + self.square_size
            elif segment == 2:  
                x = self.square_size
                y = 1.0
                z = self.z_base + self.square_size * (1 - smooth_t)
            else:             
                x = self.square_size * (1 - smooth_t)
                y = 1.0
                z = self.z_base
        
        else:
            # Static target position
            
            x = 1.0
            y = 1.0
            z = 1.0
            
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
