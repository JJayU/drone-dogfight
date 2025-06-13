#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray  
import numpy as np
import time
import sys
import select
import math

class Experiment5Node(Node):
    def __init__(self):
        super().__init__('experiment5_node')

        # Drone state
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0
        self.drone_roll = 0.0
        self.drone_pitch = 0.0
        self.drone_yaw = 0.0

        # Energy tracking
        self.motor_powers = [0.0, 0.0, 0.0, 0.0]  
        self.cumulative_energy = 0.0

        # Target trajectory parameters
        self.target_center_x = 0.0 
        self.target_center_y = 0.0  
        self.target_base_height = 2.0
        self.target_speed = 0.5
        self.figure8_radius = 3.0
        self.height_amplitude = 1.0

        # Current target state
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.target_direction_angle = 0.0

        # Following parameters
        self.follow_distance = 0.5
        self.follow_height_offset = 0.0

        self.gps_sub = self.create_subscription(PointStamped, 'gps', self.gps_callback, 10)
        self.imu_sub = self.create_subscription(Imu, 'imu', self.imu_callback, 10)
        self.motor_power_sub = self.create_subscription(
            Float32MultiArray, 
            'motor_power', 
            self.motor_power_callback, 
            10
        )
        
        # Publishers - publish desired position for control node to follow
        self.drone_des_pos_pub = self.create_publisher(PoseStamped, 'drone_des_pos', 10)
        
        # Publisher for target visualization (arrow showing target position and orientation)
        self.target_viz_pub = self.create_publisher(PoseStamped, 'target_visualization', 10)

        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.update)

        self.start_time = time.time()
        self.experiment_duration = 60.0
        self.experiment_started = False
        self.exp_no = 0

        print("\n[Eksperyment 5] Śledzenie celu i celowanie w jego tył")
        print("Cel porusza się po trajektorii ósemki z pionową falą")
        print("Dron śledzi cel i utrzymuje pozycję za nim, celując w jego tył")
        print("Naciśnij Enter, aby rozpocząć eksperyment.")

    def gps_callback(self, msg):
        self.drone_x = msg.point.x
        self.drone_y = msg.point.y
        self.drone_z = msg.point.z

    def imu_callback(self, msg):
        orientation = msg.orientation
        self.drone_roll, self.drone_pitch, self.drone_yaw = self.quaternion_to_euler(
            [orientation.w, orientation.x, orientation.y, orientation.z]
        )

    def motor_power_callback(self, msg):
        """Odbiera sygnały mocy silników i oblicza zużycie energii"""
        if len(msg.data) >= 4:
            self.motor_powers = list(msg.data[:4])
            
            power_squared_sum = sum(power**2 for power in self.motor_powers)
            energy_increment = power_squared_sum * self.dt
            
            if self.experiment_started:
                self.cumulative_energy += energy_increment

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

    def euler_to_quaternion(self, roll, pitch, yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return [w, x, y, z]

    def calculate_target_position(self, elapsed_time):
        """Oblicza pozycję celu na trajektorii ósemki z kierunkiem ruchu"""
        t = (elapsed_time * self.target_speed) / self.figure8_radius
        scale = self.figure8_radius
        
        target_x = self.target_center_x + scale * np.sin(t)
        target_y = self.target_center_y + scale * np.sin(2 * t) / 2
        
        height_variation = self.height_amplitude * np.sin(0.5 * t)
        target_z = self.target_base_height + height_variation

        dt = 0.01
        t_next = t + dt
        next_x = self.target_center_x + scale * np.sin(t_next)
        next_y = self.target_center_y + scale * np.sin(2 * t_next) / 2
        
        direction_angle = np.arctan2(next_y - target_y, next_x - target_x)

        return target_x, target_y, target_z, direction_angle


    def calculate_drone_desired_position(self):
        """Oblicza pozycję drona za celem, celującego w jego tył"""
        
        rear_angle = self.target_direction_angle + np.pi
        desired_x = self.target_x + self.follow_distance * np.cos(rear_angle)
        desired_y = self.target_y + self.follow_distance * np.sin(rear_angle)
        desired_z = self.target_z + self.follow_height_offset
        
        aim_yaw = np.arctan2(self.target_y - self.drone_y, self.target_x - self.drone_x)
        
        return desired_x, desired_y, desired_z, aim_yaw

    def log_data(self, elapsed_time):
        """Loguje dane eksperymentu"""
        distance_to_target = np.sqrt(
            (self.drone_x - self.target_x)**2 + 
            (self.drone_y - self.target_y)**2 + 
            (self.drone_z - self.target_z)**2
        )
        target_bearing = np.arctan2(self.target_y - self.drone_y, self.target_x - self.drone_x)
        aiming_error = abs(self.drone_yaw - target_bearing)
        if aiming_error > np.pi:
            aiming_error = 2 * np.pi - aiming_error

        with open(f'/home/ws/exp_data/exp5_data_{self.exp_no}.txt', 'a') as f:
            f.write(f"{elapsed_time:.3f}, "
                    f"{self.drone_x:.3f}, {self.drone_y:.3f}, {self.drone_z:.3f}, "
                    f"{self.drone_roll:.3f}, {self.drone_pitch:.3f}, {self.drone_yaw:.3f}, "
                    f"{self.target_x:.3f}, {self.target_y:.3f}, {self.target_z:.3f}, "
                    f"{self.target_direction_angle:.3f}, {distance_to_target:.3f}, "
                    f"{aiming_error:.3f}, "
                    f"{self.motor_powers[0]:.6f}, {self.motor_powers[1]:.6f}, "
                    f"{self.motor_powers[2]:.6f}, {self.motor_powers[3]:.6f}, "
                    f"{self.cumulative_energy:.6f}\n")

    def update(self):
        """Główna pętla eksperymentu"""
        desired_pos = PoseStamped()
        desired_pos.header.stamp = self.get_clock().now().to_msg()
        desired_pos.header.frame_id = 'map'

        if sys.stdin in select.select([sys.stdin], [], [], 0)[0] and not self.experiment_started:
            line = sys.stdin.readline()
            if line == '\n':
                self.experiment_started = True
                self.start_time = time.time()
                self.cumulative_energy = 0.0  
                self.exp_no += 1
                print(f"\n[Start] Eksperyment {self.exp_no} rozpoczęty.")
                
                with open(f'/home/ws/exp_data/exp5_data_{self.exp_no}.txt', 'w') as f:
                    f.write("time, drone_x, drone_y, drone_z, drone_roll, drone_pitch, drone_yaw, "
                            "target_x, target_y, target_z, target_direction, distance_to_target, "
                            "aiming_error, m1_power, m2_power, m3_power, m4_power, cumulative_energy\n")

        if self.experiment_started:
            elapsed_time = time.time() - self.start_time
            
            self.target_x, self.target_y, self.target_z, self.target_direction_angle = self.calculate_target_position(elapsed_time)
            
            target_viz = PoseStamped()
            target_viz.header.stamp = self.get_clock().now().to_msg()
            target_viz.header.frame_id = 'map'
            target_viz.pose.position.x = self.target_x
            target_viz.pose.position.y = self.target_y
            target_viz.pose.position.z = self.target_z
            
            target_quat = self.euler_to_quaternion(0.0, 0.0, self.target_direction_angle)
            target_viz.pose.orientation.w = target_quat[0]
            target_viz.pose.orientation.x = target_quat[1]
            target_viz.pose.orientation.y = target_quat[2]
            target_viz.pose.orientation.z = target_quat[3]
            
            self.target_viz_pub.publish(target_viz)
            
            des_x, des_y, des_z, des_yaw = self.calculate_drone_desired_position()
            
            desired_pos.pose.position.x = des_x
            desired_pos.pose.position.y = des_y
            desired_pos.pose.position.z = des_z
            
            quat = self.euler_to_quaternion(0.0, 0.0, des_yaw)
            desired_pos.pose.orientation.w = quat[0]
            desired_pos.pose.orientation.x = quat[1]
            desired_pos.pose.orientation.y = quat[2]
            desired_pos.pose.orientation.z = quat[3]
            
            self.log_data(elapsed_time)

            if elapsed_time > self.experiment_duration:
                print(f"[Koniec] Eksperyment zakończony. Dane zapisane do: exp5_data_{self.exp_no}.txt")
                print(f"[Energia] Całkowite zużycie energii: {self.cumulative_energy:.6f}")
                print("Cel poruszał się po ósemce, dron śledził go z tyłu.")
                print("Naciśnij Enter, aby rozpocząć kolejny.")
                self.experiment_started = False
        else:
            desired_pos.pose.position.x = 0.0
            desired_pos.pose.position.y = 0.0
            desired_pos.pose.position.z = 2.0
            desired_pos.pose.orientation.x = 0.0
            desired_pos.pose.orientation.y = 0.0
            desired_pos.pose.orientation.z = 0.0
            desired_pos.pose.orientation.w = 1.0

        self.drone_des_pos_pub.publish(desired_pos)

def main():
    rclpy.init()
    node = Experiment5Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nZamykanie programu...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()