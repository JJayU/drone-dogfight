import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Imu
import numpy as np

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
        error = self.setpoint - measured_value
        
        # Zwiększamy limit całkowania dla lepszej eliminacji błędu
        self.integral = np.clip(self.integral + error * dt, -0.5, 0.5)

        derivative = (error - self.previous_error) / max(dt, 0.001)
        # Większe filtrowanie pochodnej dla stabilności
        derivative = 0.9 * self.prev_derivative + 0.1 * derivative
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
        
        # Stan drona
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        # Parametry trybów - zwiększamy czasy
        self.current_mode = "init"  
        self.mode_switch_time = self.get_clock().now()
        self.stabilize_duration = 4.0  # Zwiększone z 2.0 na 4.0
        self.target_duration = 1.0     # Zwiększone z 0.3 na 1.0
        
        # Punkt docelowy
        self.target_point = {'x': 2.0, 'y': 2.0, 'z': 1.5}
        
        # Punkty stabilizacji
        self.stable_setpoints = {
            'x': 0.0,
            'y': 0.0,
            'z': 1.0,
            'yaw': 0.0
        }
        
        # Zwiększamy tolerancję dla bardziej stabilnego zachowania
        self.position_tolerance = 0.15  # było 0.1
        self.angle_tolerance = 0.15     # było 0.1
        
        # Subskrypcje
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
        
        # Publisher
        self.motor_pub = self.create_publisher(
            Float32MultiArray,
            'motor_power',
            10
        )
        
        # Wzmocnione regulatory PID
        self.x_pos_pid = PID(kp=0.8, ki=0.01, kd=0.8, setpoint=0.0, name="X Position")
        self.y_pos_pid = PID(kp=0.8, ki=0.01, kd=0.8, setpoint=0.0, name="Y Position")
        self.height_pid = PID(kp=2.0, ki=0.02, kd=0.6, setpoint=1.0, name="Height")
        
        # Regulatory orientacji - zwiększamy składową P i D
        self.roll_pid = PID(kp=3.0, ki=0.02, kd=0.6, setpoint=0.0, name="Roll")
        self.pitch_pid = PID(kp=3.0, ki=0.02, kd=0.6, setpoint=0.0, name="Pitch")
        self.yaw_pid = PID(kp=1.5, ki=0.1, kd=0.5, setpoint=0.0, name="Yaw")
        
        self.dt = 0.005
        self.timer = self.create_timer(self.dt, self.control_update)

    def is_at_stable_point(self):
        """Sprawdza czy dron jest w punkcie stabilnym"""
        pos_error = np.sqrt(
            (self.x - self.stable_setpoints['x'])**2 +
            (self.y - self.stable_setpoints['y'])**2 +
            (self.z - self.stable_setpoints['z'])**2
        )
        
        angle_error = abs(self.yaw - self.stable_setpoints['yaw'])
        
        # Dodajemy sprawdzenie prędkości
        vel_error = np.sqrt(
            (self.x_pos_pid.prev_derivative)**2 +
            (self.y_pos_pid.prev_derivative)**2 +
            (self.height_pid.prev_derivative)**2
        )
        
        is_stable = (pos_error < self.position_tolerance and 
                    angle_error < self.angle_tolerance and
                    vel_error < 0.1)  # Dodane sprawdzenie prędkości
        
        if is_stable:
            print(f"\n=== Position error: {pos_error:.3f}m, Angle error: {angle_error:.3f}rad, Velocity error: {vel_error:.3f}m/s ===")
        
        return is_stable
    
    def update_mode(self):
        """Aktualizuje tryb pracy na podstawie czasu i pozycji"""
        if self.current_mode == "init":
            if self.is_at_stable_point():
                self.current_mode = "stabilize"
                self.mode_switch_time = self.get_clock().now()
                print("\n=== Reached stable point, switching to STABILIZE mode ===")
                return
            return  # Pozostań w trybie init jeśli nie osiągnięto stabilizacji
            
        current_time = self.get_clock().now()
        time_in_mode = (current_time - self.mode_switch_time).nanoseconds / 1e9

        if self.current_mode == "stabilize" and time_in_mode >= self.stabilize_duration:
            if self.is_at_stable_point():  # Dodane sprawdzenie stabilności
                self.current_mode = "target"
                self.mode_switch_time = current_time
                print("\n=== Switching to TARGET mode ===")
        elif self.current_mode == "target" and time_in_mode >= self.target_duration:
            self.current_mode = "stabilize"
            self.mode_switch_time = current_time
            print("\n=== Switching to STABILIZE mode ===")
    def gps_callback(self, msg):
        self.x = msg.point.x
        self.y = msg.point.y
        self.z = msg.point.z
        
    def imu_callback(self, msg):
        q = [
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z
        ]
        self.roll, self.pitch, self.yaw = self.quaternion_to_euler(q)
        
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
        # Aktualizacja trybu
        self.update_mode()

        if self.current_mode == "init" or self.current_mode == "stabilize":
            # W trybie init i stabilize używamy standardowych PID-ów pozycji
            desired_pitch = self.x_pos_pid.update(self.x, self.dt)
            desired_roll = -self.y_pos_pid.update(self.y, self.dt)
            
            # Mniejsze ograniczenie kątów dla lepszej kontroli
            desired_pitch = np.clip(desired_pitch, -0.25, 0.25)
            desired_roll = np.clip(desired_roll, -0.25, 0.25)
            
            self.height_pid.setpoint = self.stable_setpoints['z']
            self.yaw_pid.setpoint = self.stable_setpoints['yaw']

        else:  # Tryb celowania
            target_yaw, target_pitch, target_roll = self.calculate_target_orientation()
            
            # Zwiększamy maksymalny kąt przechyłu
            max_tilt = 0.4  # około 23 stopni
            desired_pitch = np.clip(target_pitch, -max_tilt, max_tilt)
            desired_roll = np.clip(target_roll, -max_tilt, max_tilt)
            
            self.yaw_pid.setpoint = target_yaw
        
        # Ustaw setpointy dla kontrolerów orientacji
        self.pitch_pid.setpoint = desired_pitch
        self.roll_pid.setpoint = desired_roll
        
        # Oblicz sygnały sterujące
        height_control = self.height_pid.update(self.z, self.dt)
        pitch_control = self.pitch_pid.update(self.pitch, self.dt)
        roll_control = self.roll_pid.update(self.roll, self.dt)
        yaw_control = self.yaw_pid.update(self.yaw, self.dt)
        
        # Zwiększona bazowa moc silników
        base = 0.07
        
        # Mieszanie sygnałów sterujących
        m1 = base + height_control - pitch_control + roll_control - yaw_control
        m2 = base + height_control - pitch_control - roll_control + yaw_control
        m3 = base + height_control + pitch_control - roll_control - yaw_control
        m4 = base + height_control + pitch_control + roll_control + yaw_control
        
        # Normalizacja mocy silników
        max_thrust = max(abs(m1), abs(m2), abs(m3), abs(m4))
        if max_thrust > 1.0:
            m1 /= max_thrust
            m2 /= max_thrust
            m3 /= max_thrust
            m4 /= max_thrust
        
        # Publikacja komend silników
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