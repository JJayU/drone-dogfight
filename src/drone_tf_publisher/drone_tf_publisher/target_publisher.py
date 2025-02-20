#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class TargetPublisherNode(Node):
    def __init__(self):
        super().__init__('target_publisher_node')
        
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        
        # Stały punkt docelowy
        self.target_point = {'x': 2.0, 'y': 2.0, 'z': 1.5}
        
        self.target_pub = self.create_publisher(PointStamped, '/target_point', qos)
        self.create_timer(0.02, self.loop)
    
    def loop(self):
        current_time = self.get_clock().now()
        
        point_msg = PointStamped()
        point_msg.header.stamp = current_time.to_msg()
        point_msg.header.frame_id = "map"
        point_msg.point.x = self.target_point['x']
        point_msg.point.y = self.target_point['y']
        point_msg.point.z = self.target_point['z']
        
        self.target_pub.publish(point_msg)

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