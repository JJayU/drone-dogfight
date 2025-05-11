#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import Imu
import numpy as np
import time
import matplotlib.pyplot as plt

class PID:
    def __init__(self, kp, ki, kd, setpoint, name="unnamed"):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0
        self.prev_derivative = 0
        self.name = name
        
    def update(self, measured_value, dt):
        if self.name == "yaw":
            error = np.arctan2(np.sin(self.setpoint - measured_value), np.cos(self.setpoint - measured_value))
        else:
            error = self.setpoint - measured_value
        
        self.integral = np.clip(self.integral + error * dt, -5, 5)
        derivative = (error - self.previous_error) / max(dt, 0.001)
        derivative = 0.732045 * self.prev_derivative + (1-0.732045) * derivative
        self.prev_derivative = derivative
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        
        return output

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        
        # Initialize parameters
        self.declare_parameter('optitrack_topic', 'optitrack/rigid_body_0')
        self.optitrack_topic = self.get_parameter('optitrack_topic').value
        # ZMIENIC NA PRAWDZIWT TOPIC
        
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.last_time = time.time()

        # Subscribe to OptiTrack pose data
        self.optitrack_sub = self.create_subscription(
            PoseStamped,
            self.optitrack_topic,
            self.optitrack_callback,
            10
        )
        
        # Create a publisher for motor commands
        self.motor_pub = self.create_publisher(Float32MultiArray, 'motor_power', 10)

        # Initialize PID controllers
        # self.x_pos_pid = PID(kp=0.521558, ki=0.086138, kd=0.252982, setpoint=0.0, name="X Position")
        # self.y_pos_pid = PID(kp=0.662796, ki=0.088061, kd=0.375998, setpoint=0.0, name="Y Position")
        # self.roll_pid = PID(kp=0.220546, ki=0.037910, kd=0.023825, setpoint=0.0, name="Roll")
        # self.pitch_pid = PID(kp=0.220546, ki=0.037910, kd=0.023825, setpoint=0.0, name="Pitch")
        # self.yaw_pid = PID(kp=0.044144, ki=0.018837, kd=0.029852, setpoint=0.0, name="yaw")
        # self.height_pid = PID(kp=1.704485, ki=0.389549, kd=0.908491, setpoint=1.0, name="Height")
        # 
        # self.x_pos_pid = PID(kp=0.45212193, ki=0.13081939, kd=0.22335201, setpoint=0.0, name="X Position")
        # self.y_pos_pid = PID(kp=0.98237663, ki=0.11666140, kd=0.13053884, setpoint=0.0, name="Y Position")
        # self.roll_pid = PID(kp=0.18494630, ki=0.02123840, kd=0.00930142, setpoint=0.0, name="Roll")
        # self.pitch_pid = PID(kp=0.08480517, ki=0.00196951, kd=0.02869583, setpoint=0.0, name="Pitch")
        # self.yaw_pid = PID(kp=0.03604621, ki=0.01514712, kd=0.00572155, setpoint=0.0, name="yaw")
        # self.height_pid = PID(kp=1.74166147, ki=0.60216497, kd=0.83497426, setpoint=1.0, name="Height")
        
        
        self.x_pos_pid  = PID(kp=0.20, ki=0.0, kd=0.05, setpoint=0.0, name="X Position")
        self.y_pos_pid  = PID(kp=0.20, ki=0.0, kd=0.05, setpoint=0.0, name="Y Position")

        self.roll_pid   = PID(kp=0.5, ki=0.05, kd=0.1, setpoint=0.0, name="Roll")
        self.pitch_pid  = PID(kp=0.5, ki=0.05, kd=0.1, setpoint=0.0, name="Pitch")
        self.yaw_pid    = PID(kp=0.1, ki=0.05, kd=0.1, setpoint=0.0, name="Yaw")

        self.height_pid = PID(kp=1.00, ki=0.5, kd=0.50, setpoint=0.5, name="Height")

        # MOZE NEI DZIAŁC BO BYŁ TRENOWANY NA 200 a czy tyle ogarnie :)
        self.dt = 0.005  # 200Hz control loop 
        # self.timer = self.create_timer(self.dt, self.control_update)

        # For debugging
        self.drone_trajectory = []
        
        self.get_logger().info("PID Controller Node initialized")

    def optitrack_callback(self, msg):
        """Process OptiTrack pose data"""
        # DO SPRAWDZENIA CZY TAKIE COS ZWRACA
        # Extract position
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        self.z = msg.pose.position.z
        
        # Extract orientation (quaternion to euler)
        qw = msg.pose.orientation.w
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        
        # Convert quaternion to euler angles
        self.roll, self.pitch, self.yaw = self.quaternion_to_euler([qw, qx, qy, qz])
        
        # Update timestamp
        self.dt = time.time() - self.last_time
        self.last_time = time.time()
        
        self.control_update()
        
    def quaternion_to_euler(self, q):
        w, x, y, z = q
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)
        
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
        
    def control_update(self):
        """Run the PID control loop"""
        # Make sure we have received recent OptiTrack data
        if time.time() - self.last_time > 0.5:  # If data is older than 0.5 seconds
            self.get_logger().warn("No recent OptiTrack data, stopping motors")
            # Send zero motor commands for safety
            motor_commands = Float32MultiArray()
            motor_commands.data = [0.0, 0.0, 0.0, 0.0]
            self.motor_pub.publish(motor_commands)
            return
        
        # Update height PID
        height_control = self.height_pid.update(self.z, self.dt)
        
        # Calculate desired pitch and roll based on x,y position error
        desired_pitch = 0.0#self.x_pos_pid.update(self.x, self.dt)
        desired_roll = 0.0#self.y_pos_pid.update(self.y, self.dt)
        
        # Limit pitch and roll angles for safety
        desired_pitch = np.clip(desired_pitch, -0.5, 0.5)
        desired_roll = np.clip(desired_roll, -0.5, 0.5)
        
        # Account for current yaw when applying pitch and roll commands
        self.pitch_pid.setpoint = desired_pitch * np.cos(self.yaw) - desired_roll * np.sin(-self.yaw)
        self.roll_pid.setpoint = - desired_roll * np.cos(-self.yaw) + desired_pitch * np.sin(self.yaw)
        
        # Keep the drone pointing forward
        self.yaw_pid.setpoint = 0.0
        
        # Calculate final control signals
        roll_control = self.roll_pid.update(self.roll, self.dt)
        pitch_control = self.pitch_pid.update(self.pitch, self.dt)
        yaw_control = self.yaw_pid.update(self.yaw, self.dt)
        
        # Convert to motor commands
        # m1 = height_control - pitch_control + roll_control - yaw_control
        # m2 = height_control - pitch_control - roll_control + yaw_control
        # m3 = height_control + pitch_control - roll_control - yaw_control
        # m4 = height_control + pitch_control + roll_control + yaw_control
        
        m1 = - pitch_control + roll_control - yaw_control
        m2 = - pitch_control - roll_control + yaw_control
        m3 = + pitch_control - roll_control - yaw_control
        m4 = + pitch_control + roll_control + yaw_control
        
        # Normalize if any value exceeds 1.0
        max_thrust = max(abs(m1), abs(m2), abs(m3), abs(m4))
        if max_thrust > 1.0:
            m1 /= max_thrust
            m2 /= max_thrust
            m3 /= max_thrust
            m4 /= max_thrust
        
        # Ensure all values are positive
        m1 = np.clip(m1, 0.0, 1.0)
        m2 = np.clip(m2, 0.0, 1.0)
        m3 = np.clip(m3, 0.0, 1.0)
        m4 = np.clip(m4, 0.0, 1.0)
        
        # Publish motor commands
        motor_commands = Float32MultiArray()
        motor_commands.data = [float(m1), float(m2), float(m3), float(m4)]
        self.motor_pub.publish(motor_commands)
        
        # # Log drone position for debugging
        # self.drone_trajectory.append([self.x, self.y, self.z])
        
        # Debug output
        if len(self.drone_trajectory) % 100 == 0:  # Log every 100 iterations
            # self.get_logger().info(f"Position: x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f}")
            # self.get_logger().info(f"Motors: m1={m1:.2f}, m2={m2:.2f}, m3={m3:.2f}, m4={m4:.2f}")
            # self.get_logger().info(f"PID Outputs: height={height_control:.2f}, pitch={pitch_control:.2f}, roll={roll_control:.2f}, yaw={yaw_control:.2f}")
            self.get_logger().info(f"Orientation: roll={self.roll:.2f}, pitch={self.pitch:.2f}, yaw={self.yaw:.2f}")

    def plot_trajectories(self):
        """Save trajectory plot for debugging"""
        drone_x = [pos[0] for pos in self.drone_trajectory]
        drone_y = [pos[1] for pos in self.drone_trajectory]
        drone_z = [pos[2] for pos in self.drone_trajectory]
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(drone_x, drone_y, drone_z, label='Drone Trajectory', color='red')
        ax.set_title('Drone Trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('drone_trajectory_plot.png')
        self.get_logger().info("Trajectory plot saved to drone_trajectory_plot.png")

def main():
    rclpy.init()
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # node.plot_trajectories()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()