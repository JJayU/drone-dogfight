import mujoco
import mujoco_viewer
import numpy as np
import rclpy
from rclpy.node import Node
import os
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Imu
from ament_index_python.packages import get_package_share_directory
import cffirmware as firm
import math
# from simple_pid import PID
class PID:
    def __init__(self, kp, ki, kd, setpoint, name="unnamed"):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0
        self.name = name  # Dodajemy nazwę dla identyfikacji kontrolera
        
    def update(self, measured_value, dt):
        error = self.setpoint - measured_value
        self.integral += error * dt
        # self.integral = np.clip(self.integral, -self.windup_limit, self.windup_limit)
        derivative = (error - self.previous_error) / dt
        
        # Obliczamy poszczególne składniki PID
        p_term = self.kp * error
        i_term = self.ki * self.integral
        d_term = self.kd * derivative
        
        output = p_term + i_term + d_term
        self.previous_error = error
        
        # Wyświetlamy informacje diagnostyczne
        print(f"\n=== {self.name} PID Debug ===")
        print(f"Setpoint: {self.setpoint:.4f}")
        print(f"Measured: {measured_value:.4f}")
        print(f"Error: {error:.4f}")
        print(f"P term ({self.kp}): {p_term:.4f}")
        print(f"I term ({self.ki}): {i_term:.4f}")
        print(f"D term ({self.kd}): {d_term:.4f}")
        print(f"Integral: {self.integral:.4f}")
        print(f"Derivative: {derivative:.4f}")
        print(f"Output: {output:.4f}")
        print("=====================")
        
        return output
        
class SimNode(Node):
    def __init__(self):
        super().__init__('sim_node')

        package_share_dir = get_package_share_directory('drone_mujoco')
        model_path = os.path.join(package_share_dir, 'model', 'scene.xml')
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

        self.model.opt.timestep = 0.005
        self.create_timer(0.005, self.loop)
        self.running = True

        self.motor_subscription = self.create_subscription(
            Float32MultiArray,
            'motor_power',
            self.motor_listener_callback,
            10
        )

        self.gps_publisher = self.create_publisher(PointStamped, 'gps', 10)
        self.imu_publisher = self.create_publisher(Imu, 'imu', 10)

        # Kontrolery pozycji (zewnętrzne)
        self.x_pos_pid = PID(kp=2.0, ki=0.01, kd=0.5, setpoint=0.0, name="X Position")
        self.y_pos_pid = PID(kp=2.0, ki=0.01, kd=0.5, setpoint=1.0, name="Y Position")
        
        # Kontrolery orientacji (wewnętrzne)
        self.height_pid = PID(kp=10.0, ki=0.05, kd=1.5, setpoint=1.0)
        self.roll_pid = PID(kp=15.0, ki=0.05, kd=1.0, setpoint=0.0)
        self.pitch_pid = PID(kp=15.0, ki=0.05, kd=1.0, setpoint=0.0)
        self.yaw_pid = PID(kp=0.1, ki=0.1, kd=0.5, setpoint=1.0)
        

    def motor_listener_callback(self, msg):
        if len(msg.data) == 4:
            self.set_control(*msg.data)
        else:
            self.get_logger().warn('Invalid motor command length!')

    def set_control(self, m1, m2, m3, m4):
        # Clamp motor values to prevent negative thrust
        self.data.ctrl[0] = np.clip(m1, 0, 1)
        self.data.ctrl[1] = np.clip(m2, 0, 1)
        self.data.ctrl[2] = np.clip(m3, 0, 1)
        self.data.ctrl[3] = np.clip(m4, 0, 1)

    def publish_state(self):
        x, y, z = self.data.qpos[0:3]

        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg() 
        msg.header.frame_id = "map"
        msg.point.x, msg.point.y, msg.point.z = x, y, z

        self.gps_publisher.publish(msg)

        quaternion = self.data.qpos[3:7]

        msg2 = Imu()
        msg2.orientation.w = quaternion[0]
        msg2.orientation.x = quaternion[1]
        msg2.orientation.y = quaternion[2]
        msg2.orientation.z = quaternion[3]

        self.imu_publisher.publish(msg2)
    def quaternion_to_euler(self, quaternion):
        w, x, y, z = quaternion
        
        # Roll (obrót wokół osi X)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (obrót wokół osi Y)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        # Yaw (obrót wokół osi Z)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
    def loop(self):
        if self.viewer.is_alive:
            dt = self.model.opt.timestep

            # Pomiary stanu
            x, y, z = self.data.qpos[0:3]
            quaternion = self.data.qpos[3:7]
            roll, pitch, yaw = self.quaternion_to_euler(quaternion)
            vx, vy, vz = self.data.qvel[0:3]
            
            # 1. Kontrolery pozycji generują zadane kąty
            # Zadany pitch jest proporcjonalny do błędu pozycji X
            desired_pitch = self.x_pos_pid.update(x, dt)  # minus bo pitch do przodu (ujemny) powoduje ruch w kierunku +X
            # Zadany roll jest proporcjonalny do błędu pozycji Y
            desired_roll = -self.y_pos_pid.update(y, dt)        
            # Ograniczenie maksymalnych kątów dla bezpieczeństwa (np. 15 stopni = 0.2 rad)
            desired_pitch = np.clip(desired_pitch, -0.2, 0.2)
            desired_roll = np.clip(desired_roll, -0.2, 0.2)
            
            # 2. Kontrolery orientacji stabilizują kąty
            self.pitch_pid.setpoint = desired_pitch
            self.roll_pid.setpoint =  desired_roll
            
            height_control = self.height_pid.update(z, dt)
            pitch_control = self.pitch_pid.update(pitch, dt) 
            roll_control = self.roll_pid.update(roll, dt) 
            yaw_control = self.yaw_pid.update(yaw, dt) 
            base = 2
            # Mieszanie sygnałów sterujących dla silników (układ X)
            m1 = base + height_control - pitch_control + roll_control - yaw_control   # przód prawy
            m2 = base + height_control - pitch_control - roll_control + yaw_control   # tył prawy
            m3 = base + height_control + pitch_control - roll_control - yaw_control   # tył lewy
            m4 = base + height_control + pitch_control + roll_control + yaw_control   # przód lewy

            # Normalizacja i ograniczenie sygnałów sterujących
            max_thrust = max(abs(m1), abs(m2), abs(m3), abs(m4))
            if max_thrust > 1.0:
                m1 /= max_thrust
                m2 /= max_thrust
                m3 /= max_thrust
                m4 /= max_thrust

            self.set_control(m1, m2, m3, m4)
            mujoco.mj_step(self.model, self.data)
            self.viewer.render()
            self.publish_state()
        else:
            self.running = False
            self.destroy_node()

def main():
    rclpy.init()
    node = SimNode()
    try:
        while rclpy.ok() and node.running:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()