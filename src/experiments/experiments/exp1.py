import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import Imu
import time
import math
import os
import sys
import select

class Experiment3Node(Node):
    def __init__(self):
        super().__init__('experiment3_node')
        
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0
        self.drone_roll = 0.0
        self.drone_pitch = 0.0
        self.drone_yaw = 0.0
        
        # Pozycja domowa (home position)
        self.home_x = 0.0
        self.home_y = 0.0
        self.home_z = 1.0
        self.home_yaw = 0.0
        
        self.gps_sub = self.create_subscription(PointStamped, 'gps', self.gps_callback, 10)
        self.imu_sub = self.create_subscription(Imu, 'imu', self.imu_callback, 10)
        self.drone_des_pos_pub = self.create_publisher(PoseStamped, 'drone_des_pos', 10)
        
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.update)
        
        self.experiment_duration = 30.0
        self.hit_duration = 2.0
        self.hit_threshold = 0.3
        
        # 16 stałych punktów z orientacją (x, y, z, yaw)
# 15 nowych celów z nowymi pozycjami i odwrotnymi yaw (pierwotny_yaw + π)
        self.defined_targets = [
            [1.0, 1.0, 1.5, 0.0],
            [-1.0, 1.0, 1.0, 0.5],
            [2.0, -1.5, 2.0, 1.0],
            [-2.0, -1.5, 1.8, 1.5],
            [0.0, 2.5, 1.0, -1.0],
            [1.5, -2.5, 2.2, -0.5],
            [-1.5, 2.0, 1.4, 0.8],
            [2.5, 0.0, 1.9, 0.2],
            [-2.5, 0.0, 2.1, -0.8],
            [0.0, -2.5, 1.3, -1.2],
            [2.2, 2.2, 2.0, 1.57],
            [-2.2, -2.2, 1.2, -1.57],
            [1.0, -1.0, 1.7, 3.14],
            [-1.0, 0.0, 1.5, -3.14],
            [0.0, 0.0, 2.0, 0.0],
            [1.5, 1.5, 1.6, 0.3],
        ]
        self.current_target_index = 0
        self.current_target = None
        self.target_hit_start_time = None
        self.is_hitting_target = False
        
        self.start_time = None
        self.experiment_started = False
        self.experiment_finished = False
        self.exp_no = 0
        self.targets_hit = 0
        self.total_targets = 0
        
        self.data_dir = "/home/ws/exp_data"
        self.data_file = None
        self.setup_data_logging()
        
        print("\nExperiment 3 Node initialized - Static Target Challenge")
        print("Drone has 30 seconds to hit as many fixed targets as possible")
        print("Drone will start from home position and return home after experiment")
        print("Press Enter in terminal to start...")

    def setup_data_logging(self):
        """Setup data logging directory and file"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created data directory: {self.data_dir}")

    def start_data_logging(self):
        """Start logging data to file"""
        filename = f"exp3_data_{self.exp_no}.txt"
        self.data_file_path = os.path.join(self.data_dir, filename)
        self.data_file = open(self.data_file_path, 'w')
        
        # Dodany yaw_error do nagłówka
        header = "time,targets_hit,total_targets,drone_x,drone_y,drone_z,drone_roll,drone_pitch,drone_yaw,target_x,target_y,target_z,target_yaw,distance_to_target,yaw_error,is_hitting\n"
        self.data_file.write(header)
        print(f"Started logging to: {self.data_file_path}")

    def log_data(self, elapsed_time):
        """Log current data to file"""
        if self.data_file and self.current_target:
            distance = self.calculate_distance_to_target()
            yaw_error = self.calculate_yaw_error()
            is_hitting = 1 if self.is_hitting_target else 0
            
            data_line = f"{elapsed_time:.3f},{self.targets_hit},{self.total_targets}," \
                       f"{self.drone_x:.3f},{self.drone_y:.3f},{self.drone_z:.3f}," \
                       f"{self.drone_roll:.3f},{self.drone_pitch:.3f},{self.drone_yaw:.3f}," \
                       f"{self.current_target[0]:.3f},{self.current_target[1]:.3f},{self.current_target[2]:.3f}," \
                       f"{self.current_target[3]:.3f},{distance:.3f},{yaw_error:.3f},{is_hitting}\n"
            
            self.data_file.write(data_line)

    def stop_data_logging(self):
        """Stop logging and close file"""
        if self.data_file:
            self.data_file.close()
            self.data_file = None
            print(f"Data saved to: {self.data_file_path}")

    def gps_callback(self, msg):
        self.drone_x = msg.point.x
        self.drone_y = msg.point.y
        self.drone_z = msg.point.z
        
    def imu_callback(self, msg):
        o = msg.orientation
        self.drone_roll, self.drone_pitch, self.drone_yaw = self.quaternion_to_euler([o.w, o.x, o.y, o.z])

    def quaternion_to_euler(self, q):
        w, x, y, z = q
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x**2 + y**2)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = math.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y**2 + z**2)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def euler_to_quaternion(self, roll, pitch, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return [w, x, y, z]

    def next_target(self):
        if self.current_target_index >= len(self.defined_targets):
            self.current_target_index = 0
        target = self.defined_targets[self.current_target_index]
        self.current_target_index += 1
        self.total_targets += 1
        print(f"New target: {target} (Total: {self.total_targets})")
        return target

    def calculate_distance_to_target(self):
        if self.current_target is None:
            return float('inf')
        dx = self.drone_x - self.current_target[0]
        dy = self.drone_y - self.current_target[1]
        dz = self.drone_z - self.current_target[2]
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def calculate_yaw_error(self):
        """Oblicz błąd yaw do aktualnego celu"""
        if self.current_target is None:
            return float('inf')
        return abs(self.wrap_angle(self.current_target[3] - self.drone_yaw))

    def wrap_angle(self, angle):
        """Normalizuj kąt do zakresu [-pi, pi]"""
        import math
        return math.atan2(math.sin(angle), math.cos(angle))

    def check_target_hit(self):
        distance = self.calculate_distance_to_target()
        yaw_error = self.calculate_yaw_error()
        now = time.time()
        
        # Sprawdź czy dron jest wystarczająco blisko w pozycji I orientacji
        position_ok = distance <= self.hit_threshold
        orientation_ok = yaw_error <= 0.1  # threshold dla yaw = 0.1 rad (~5.7 stopni)
        
        if position_ok and orientation_ok:
            if not self.is_hitting_target:
                self.is_hitting_target = True
                self.target_hit_start_time = now
                print(f"Targeting... (distance: {distance:.2f}m, yaw_error: {yaw_error:.2f}rad)")
            elif now - self.target_hit_start_time >= self.hit_duration:
                self.targets_hit += 1
                print(f"TARGET HIT! Total: {self.targets_hit}")
                self.current_target = self.next_target()
                self.is_hitting_target = False
        else:
            if self.is_hitting_target:
                print(f"Lost target (distance: {distance:.2f}m, yaw_error: {yaw_error:.2f}rad)")
                self.is_hitting_target = False

    def send_home_position(self):
        """Wyślij pozycję domową jako cel"""
        desired_pos = PoseStamped()
        desired_pos.header.stamp = self.get_clock().now().to_msg()
        desired_pos.header.frame_id = 'map'
        
        desired_pos.pose.position.x = self.home_x
        desired_pos.pose.position.y = self.home_y
        desired_pos.pose.position.z = self.home_z
        
        q = self.euler_to_quaternion(0, 0, self.home_yaw)
        desired_pos.pose.orientation.w = q[0]
        desired_pos.pose.orientation.x = q[1]
        desired_pos.pose.orientation.y = q[2]
        desired_pos.pose.orientation.z = q[3]
        
        self.drone_des_pos_pub.publish(desired_pos)

    def update(self):
        desired_pos = PoseStamped()
        desired_pos.header.stamp = self.get_clock().now().to_msg()
        desired_pos.header.frame_id = 'map'

        # Sprawdź czy użytkownik nacisnął Enter (bez blokowania)
        if not self.experiment_started and not self.experiment_finished:
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                if line == '\n':
                    self.experiment_started = True
                    self.start_time = time.time()
                    self.exp_no += 1
                    self.targets_hit = 0
                    self.total_targets = 0
                    self.current_target_index = 0
                    self.current_target = self.next_target()
                    self.start_data_logging()
                    print(f"Experiment {self.exp_no} started.")
            
            # Przed eksperymentem - zostań w pozycji domowej
            if not self.experiment_started:
                self.send_home_position()
                return

        # Podczas eksperymentu
        if self.experiment_started and not self.experiment_finished:
            elapsed = time.time() - self.start_time
            
            if elapsed >= self.experiment_duration:
                print(f"\nExperiment complete.")
                print(f"Targets hit: {self.targets_hit}")
                print(f"Total targets: {self.total_targets}")
                if self.total_targets > 0:
                    success_rate = (self.targets_hit / self.total_targets) * 100
                    print(f"Success rate: {success_rate:.1f}%")
                print("Returning to home position...")
                print("Press Enter to restart.")
                
                self.stop_data_logging()
                self.experiment_started = False
                self.experiment_finished = True
                
                # Wróć do pozycji domowej po eksperymencie
                self.send_home_position()
                return

            # Sprawdź trafienia w cel
            self.check_target_hit()
            
            # Loguj dane
            self.log_data(elapsed)

            # Wyślij aktualny cel
            desired_pos.pose.position.x = self.current_target[0]
            desired_pos.pose.position.y = self.current_target[1]
            desired_pos.pose.position.z = self.current_target[2]

            q = self.euler_to_quaternion(0, 0, self.current_target[3])
            desired_pos.pose.orientation.w = q[0]
            desired_pos.pose.orientation.x = q[1]
            desired_pos.pose.orientation.y = q[2]
            desired_pos.pose.orientation.z = q[3]

            self.drone_des_pos_pub.publish(desired_pos)

        elif self.experiment_finished:
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                if line == '\n':
                    self.experiment_finished = False
                    print("Ready for next experiment. Press Enter to start...")
            
            self.send_home_position()

def main():
    rclpy.init()
    node = Experiment3Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.stop_data_logging()  
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()