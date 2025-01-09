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
        
        self.initial_pos = None
        self.drone_pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.current_yaw = 0.0
        
        self.points = [
            {'x': 0.2, 'y': 0.0, 'z': 0.5},
            {'x': 0.1, 'y': 0.17, 'z': 0.8},
            {'x': -0.1, 'y': 0.17, 'z': 1.0},
            {'x': -0.2, 'y': 0.0, 'z': 1.2},
            {'x': -0.1, 'y': -0.17, 'z': 0.9},
            {'x': 0.1, 'y': -0.17, 'z': 0.6}
        ]
        
        self.current_point_idx = 0
        self.point_achieved_time = None
        self.point_wait_time = 2.0
        
        # Zmniejsz wzmocnienie PID
        self.pid = {
            'yaw': {'kp': 10.0, 'ki': 0.0, 'kd': 3.5, 'integral': 0.0, 'prev_error': 0.0},
            'z': {'kp': 1.0, 'ki': 0.001, 'kd': 0.1, 'integral': 0.0, 'prev_error': 0.0}
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

    def is_yaw_achieved(self, yaw_error):
        return abs(yaw_error) < self.yaw_threshold

    def control_loop(self):
        current_target = self.points[self.current_point_idx]
        
        # Oblicz kąt do celu w zakresie -pi do pi
        target_yaw = -math.atan2(current_target['x'] - self.drone_pos['x'] , current_target['y'] - self.drone_pos['y']) + math.pi/2
        yaw_error = self.normalize_angle(target_yaw - self.current_yaw)
        
        if self.is_yaw_achieved(yaw_error):
            if self.point_achieved_time is None:
                self.point_achieved_time = self.get_clock().now()
                self.get_logger().info(f'Reached point {self.current_point_idx + 1} orientation!')
            elif (self.get_clock().now() - self.point_achieved_time).nanoseconds / 1e9 >= self.point_wait_time:
                self.current_point_idx = (self.current_point_idx + 1) % len(self.points)
                self.point_achieved_time = None
                self.get_logger().info(f'Moving to point {self.current_point_idx + 1}')
        else:
            self.point_achieved_time = None
        
        # Wyświetl aktualne wartości dla debugowania
        self.get_logger().info(f"""
            Current yaw: {math.degrees(self.current_yaw):.1f}°
            Target yaw: {math.degrees(target_yaw):.1f}°
            Error: {math.degrees(yaw_error):.1f}°
            Current point: {self.current_point_idx + 1}
            Target: x={current_target['x']:.2f}, y={current_target['y']:.2f}
        """)

        # Oblicz i wyślij komendy sterujące
        cmd_vel = Twist()
        dz = current_target['z'] - self.drone_pos['z']
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