import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Imu
import numpy as np
import time

class PID:
    def __init__(self, kp, ki, kd, setpoint, name="unnamed"):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0
        self.name = name
        self.prev_derivative = 0
        
    def update(self, measured_value, dt):
        
        if self.name == "yaw":
            error = np.arctan2(np.sin(self.setpoint - measured_value), np.cos(self.setpoint - measured_value))
        else:
            error = self.setpoint - measured_value
        
        self.integral = np.clip(self.integral + error * dt, -5, 5)

        derivative = (error - self.previous_error) / max(dt, 0.001)
        derivative = 0.8 * self.prev_derivative + 0.2 * derivative
        self.prev_derivative = derivative
        
        p_term = self.kp * error
        i_term = self.ki * self.integral
        d_term = self.kd * derivative
        
        output = p_term + i_term + d_term
        self.previous_error = error
        
        return output

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        self.gps_sub = self.create_subscription(
            PointStamped,
            'gps',
            self.gps_callback,
            10
        )
        self.imu_sub = self.create_subscription(
            Imu,
            'imu',
            self.imu_callback,
            10
        )
        self.target_sub = self.create_subscription(
            PointStamped,
            'target_point',
            self.target_callback,
            10
        )
        
        self.motor_pub = self.create_publisher(
            Float32MultiArray,
            'motor_power',
            10
        )
        
        self.x_pos_pid  = PID(kp=0.20, ki=0.0, kd=0.05, setpoint=0.0, name="X Position")
        self.y_pos_pid  = PID(kp=0.20, ki=0.0, kd=0.05, setpoint=0.0, name="Y Position")

        self.roll_pid   = PID(kp=0.05, ki=0.0, kd=0.01, setpoint=0.0)
        self.pitch_pid  = PID(kp=0.05, ki=0.0, kd=0.01, setpoint=0.0)
        self.yaw_pid    = PID(kp=0.01, ki=0.0, kd=0.01, setpoint=0.0, name="yaw")

        self.height_pid = PID(kp=1.00, ki=0.5, kd=0.50, setpoint=1.0)
        
        self.dt = 0.005
        self.timer = self.create_timer(self.dt, self.control_update)
        
        self.last_time = 0.0
        
        self.target = [0., 0., 0.]
        
        self.angle = 0.
        
    def gps_callback(self, msg):
        self.x = msg.point.x
        self.y = msg.point.y
        self.z = msg.point.z
        
        self.last_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1000000000.
        
    def imu_callback(self, msg):
        q = [
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z
        ]
        self.roll, self.pitch, self.yaw = self.quaternion_to_euler(q)
        
    def target_callback(self, msg):
        self.target = [msg.point.x, msg.point.y, msg.point.z]
        
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
        
        if(time.time() - self.last_time < 1):
            
            self.angle += 0.001
            
            target_x = np.sin(self.angle)
            target_y = np.cos(self.angle)
            
            self.x_pos_pid.setpoint = target_x
            self.y_pos_pid.setpoint = target_y
            
            target_yaw = np.arctan2(self.target[1] - self.y, self.target[0] - self.x)
            
            self.height_pid.setpoint = self.target[2]
            
            print(f"{target_x} {target_y} {target_yaw}")
        
            desired_pitch = self.x_pos_pid.update(self.x, self.dt)
            desired_roll = self.y_pos_pid.update(self.y, self.dt)

            desired_pitch = np.clip(desired_pitch, -0.3, 0.3) 
            desired_roll = np.clip(desired_roll, -0.3, 0.3) 
            
            self.pitch_pid.setpoint = desired_pitch * np.cos(self.yaw) - desired_roll * np.sin(-self.yaw)
            self.roll_pid.setpoint = - desired_roll * np.cos(-self.yaw) + desired_pitch * np.sin(self.yaw)
            self.yaw_pid.setpoint = target_yaw
            
            height_control = self.height_pid.update(self.z, self.dt)
            pitch_control = self.pitch_pid.update(self.pitch, self.dt)
            roll_control = self.roll_pid.update(self.roll, self.dt)
            yaw_control = self.yaw_pid.update(self.yaw, self.dt)
            
            m1 = height_control - pitch_control + roll_control - yaw_control
            m2 = height_control - pitch_control - roll_control + yaw_control
            m3 = height_control + pitch_control - roll_control - yaw_control
            m4 = height_control + pitch_control + roll_control + yaw_control
            
            max_thrust = max(abs(m1), abs(m2), abs(m3), abs(m4))
            if max_thrust > 1.0:
                m1 /= max_thrust
                m2 /= max_thrust
                m3 /= max_thrust
                m4 /= max_thrust
            
            motor_commands = Float32MultiArray()
            motor_commands.data = [float(m1), float(m2), float(m3), float(m4)]
            self.motor_pub.publish(motor_commands)
        
def main():
    rclpy.init()
    node = ControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()