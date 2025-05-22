import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import Imu
import numpy as np
import time
import sys
import select

class Experiment1Node(Node):
    def __init__(self):
        super().__init__('experiments_node')
        
        # Variables to store drone state
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0
        self.drone_roll = 0.0
        self.drone_pitch = 0.0
        self.drone_yaw = 0.0
        
        # Subscribers and publishers
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
        self.drone_des_pos_pub = self.create_publisher(
            PoseStamped,
            'drone_des_pos',
            10
        )
        
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.update)
        
        self.start_time = time.time()
        self.experiment_duration = 5.0 
        self.experiment_started = False
        self.exp_no = 0
        
        print("\nExperiment 1 Node initialized.")
        print("Press Enter to start the experiment.")
        
    def gps_callback(self, msg):
        self.drone_x = msg.point.x
        self.drone_y = msg.point.y
        self.drone_z = msg.point.z
        
    def imu_callback(self, msg):
        orientation = msg.orientation
        self.drone_roll, self.drone_pitch, self.drone_yaw = self.quaternion_to_euler(
            [orientation.w, orientation.x, orientation.y, orientation.z]
        )
        
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
        
    def log_data(self):
        # Log the data to a file
        with open(f'/home/ws/exp_data/exp1_data_{self.exp_no}.txt', 'a') as f:
            f.write(f"{time.time() - self.start_time}, {self.drone_x}, {self.drone_y}, {self.drone_z}, {self.drone_roll}, {self.drone_pitch}, {self.drone_yaw}\n")
            
            
    def update(self):
        
        # Create a PoseStamped message for the desired position
        desired_pos = PoseStamped()
        desired_pos.header.stamp = self.get_clock().now().to_msg()
        desired_pos.header.frame_id = 'map'
        
        # Check if Enter is pressed to start the experiment
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0] and not self.experiment_started:
            line = sys.stdin.readline()
            if line == '\n':
                self.experiment_started = True
                self.start_time = time.time()
                self.exp_no += 1
                print(f"Experiment no. {self.exp_no} started!")
        
        if self.experiment_started:
            self.log_data()
            
            # Experiment logic
            desired_pos.pose.position.x = 0.0
            desired_pos.pose.position.y = 0.0
            desired_pos.pose.position.z = 1.5
            desired_pos.pose.orientation.x = 0.0
            desired_pos.pose.orientation.y = 0.0
            desired_pos.pose.orientation.z = 0.0
            desired_pos.pose.orientation.w = 1.0
            
            # Check if the experiment duration has passed
            if time.time() - self.start_time > self.experiment_duration:
                print("Experiment finished!")
                print("Press Enter to start again.")
                self.experiment_started = False
            
        else:
            # Standby position
            desired_pos.pose.position.x = 0.0
            desired_pos.pose.position.y = 0.0
            desired_pos.pose.position.z = 0.2
            desired_pos.pose.orientation.x = 0.0
            desired_pos.pose.orientation.y = 0.0
            desired_pos.pose.orientation.z = 0.0
            desired_pos.pose.orientation.w = 1.0
            
        # Publish the desired position
        self.drone_des_pos_pub.publish(desired_pos)
        
        
def main():
    rclpy.init()
    node = Experiment1Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()