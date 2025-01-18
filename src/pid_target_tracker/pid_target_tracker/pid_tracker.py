#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped
from sensor_msgs.msg import Imu
from tf_transformations import euler_from_quaternion
import math
import numpy as np

class VirtualTargetTracker(Node):
    def __init__(self):
        super().__init__('virtual_target_tracker')

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.gps_sub = self.create_subscription(PointStamped, '/crazyflie_1/gps', self.gps_callback, 10)
        self.target_sub = self.create_subscription(PointStamped, '/target_point', self.target_callback, 10)

        self.initial_pos = None
        self.drone_pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.current_yaw = 0.0

        self.target_pos = None
        
        self.point_achieved_time = None
        self.point_wait_time = 2.0
        
        # Zmniejsz wzmocnienie PID
        self.pid = {
            'yaw': {'kp': 50.0, 'ki': 0.0, 'kd': 8.5, 'integral': 0.0, 'prev_error': 0.0},
            'z': {'kp': 1.0, 'ki': 0.000, 'kd': 0.1, 'integral': 0.0, 'prev_error': 0.0}
        }
        
        self.yaw_threshold = math.radians(5)
        self.create_timer(0.02, self.control_loop)

    def imu_callback(self, msg: Imu):
        orientation = msg.orientation
        _, _, yaw = euler_from_quaternion([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ])
        self.current_yaw = yaw

    def gps_callback(self, msg: PointStamped):
        if self.initial_pos is None:
            self.initial_pos = {'x': msg.point.x, 'y': msg.point.y, 'z': msg.point.z}
        self.drone_pos['z'] = msg.point.z
        self.drone_pos['x'] = msg.point.x
        self.drone_pos['y'] = msg.point.y


    def calculate_pid(self, pid_values, error, dt=0.02):
        p_out = pid_values['kp'] * error
        pid_values['integral'] = np.clip(pid_values['integral'] + error * dt, -0.5, 0.5)
        i_out = pid_values['ki'] * pid_values['integral']
        d_out = pid_values['kd'] * (error - pid_values['prev_error']) / dt
        pid_values['prev_error'] = error
        
        total_out = np.clip(p_out + i_out + d_out, -10., 10.)
        if pid_values == self.pid['yaw']:
            total_out = np.clip(total_out, -10.0, 10.0)
        return total_out

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def target_callback(self, msg: PointStamped):
        self.target_pos = {'x': msg.point.x, 'y': msg.point.y, 'z': msg.point.z}
        self.get_logger().info(f"Received new target: x={msg.point.x}, y={msg.point.y}, z={msg.point.z}")

    def control_loop(self):

        if self.target_pos is None:
            self.get_logger().info("No target received yet.")
            return

        target_yaw = math.atan2(self.target_pos['y'] - self.drone_pos['y'],
                                self.target_pos['x'] - self.drone_pos['x'])
        yaw_error = self.normalize_angle(target_yaw - self.current_yaw)
        
        # Wyświetl aktualne wartości dla debugowania
        self.get_logger().info(f"""
            Current yaw: {math.degrees(self.current_yaw):.1f}°
            Target yaw: {math.degrees(target_yaw):.1f}°
            Yaw error: {math.degrees(yaw_error):.1f}°
            Target position: x={self.target_pos['x']:.2f}, y={self.target_pos['y']:.2f}, z={self.target_pos['z']:.2f}
        """)

        # Oblicz i wyślij komendy sterujące
        cmd_vel = Twist()
        dz = self.target_pos['z'] - self.drone_pos['z']
        cmd_vel.linear.z = float(self.calculate_pid(self.pid['z'], dz))
        cmd_vel.angular.z = float(self.calculate_pid(self.pid['yaw'], yaw_error))
        
        self.cmd_vel_pub.publish(cmd_vel)

def main():
    rclpy.init()
    tracker = VirtualTargetTracker()
    try:
        rclpy.spin(tracker)
    except KeyboardInterrupt:
        pass
    tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()