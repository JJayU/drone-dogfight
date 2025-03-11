import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Imu
import numpy as np
import time
import signal
import sys
import atexit

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
        
        # For logging
        self.error_history = []
        self.setpoint_history = []
        self.output_history = []
        self.time_history = []
        self.start_time = time.time()
        
    def update(self, measured_value, dt):
        current_time = time.time() - self.start_time
        
        if self.name == "yaw":
            error = np.arctan2(np.sin(self.setpoint - measured_value), np.cos(self.setpoint - measured_value))
        else:
            error = self.setpoint - measured_value
        
        self.integral = np.clip(self.integral + error * dt, -5, 5)

        derivative = (error - self.previous_error) / max(dt, 0.001)
        derivative = 0.682045 * self.prev_derivative + (1-0.682045) * derivative
        self.prev_derivative = derivative
        
        p_term = self.kp * error
        i_term = self.ki * self.integral
        d_term = self.kd * derivative
        
        output = p_term + i_term + d_term
        self.previous_error = error
        
        # Log data
        self.error_history.append(error)
        self.setpoint_history.append(self.setpoint)
        self.output_history.append(output)
        self.time_history.append(current_time)
        
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
        self.shutdown_requested = False
        
        # For trajectory logging
        self.position_history = {'x': [], 'y': [], 'z': [], 'time': []}
        self.attitude_history = {'roll': [], 'pitch': [], 'yaw': [], 'time': []}
        self.target_history = {'x': [], 'y': [], 'z': [], 'yaw': [], 'time': []}
        self.start_time = time.time()
        
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
        
        self.motor_pub = self.create_publisher(
            Float32MultiArray,
            'motor_power',
            10
        )
        
        self.x_pos_pid  = PID(kp=0.521558, ki=0.086138, kd=0.252982, setpoint=0.0, name="X Position")
        self.y_pos_pid  = PID(kp=0.662796, ki=0.088061, kd=0.375998, setpoint=0.0, name="Y Position")

        self.roll_pid   = PID(kp=0.220546, ki=0.037910, kd=0.023825, setpoint=0.0, name="Roll")
        self.pitch_pid  = PID(kp=0.092955, ki=0.031571, kd=0.030294, setpoint=0.0, name="Pitch")
        self.yaw_pid    = PID(kp=0.044144, ki=0.018837, kd=0.029852, setpoint=0.0, name="yaw")

        self.height_pid = PID(kp=1.704485, ki=0.389549, kd=0.908491, setpoint=1.0, name="Height")

        self.dt = 0.005
        self.timer = self.create_timer(self.dt, self.control_update)
        
        self.last_time = 0.0
        self.angle = 0.
        
        # Register for safe shutdown
        atexit.register(self.save_data)
        
    def save_data(self):
        """Save flight data to files for plotting later"""
        import pickle
        
        # If we've recorded any data, save it
        if len(self.position_history['time']) > 0:
            self.get_logger().info("Saving flight data for later plotting...")
            
            # Compile all data into one dictionary
            flight_data = {
                'position': self.position_history,
                'attitude': self.attitude_history,
                'target': self.target_history,
                'x_pid': {
                    'error': self.x_pos_pid.error_history,
                    'setpoint': self.x_pos_pid.setpoint_history,
                    'output': self.x_pos_pid.output_history,
                    'time': self.x_pos_pid.time_history,
                    'name': self.x_pos_pid.name
                },
                'y_pid': {
                    'error': self.y_pos_pid.error_history,
                    'setpoint': self.y_pos_pid.setpoint_history,
                    'output': self.y_pos_pid.output_history,
                    'time': self.y_pos_pid.time_history,
                    'name': self.y_pos_pid.name
                },
                'z_pid': {
                    'error': self.height_pid.error_history,
                    'setpoint': self.height_pid.setpoint_history,
                    'output': self.height_pid.output_history,
                    'time': self.height_pid.time_history,
                    'name': self.height_pid.name
                },
                'roll_pid': {
                    'error': self.roll_pid.error_history,
                    'setpoint': self.roll_pid.setpoint_history,
                    'output': self.roll_pid.output_history,
                    'time': self.roll_pid.time_history,
                    'name': self.roll_pid.name
                },
                'pitch_pid': {
                    'error': self.pitch_pid.error_history,
                    'setpoint': self.pitch_pid.setpoint_history,
                    'output': self.pitch_pid.output_history,
                    'time': self.pitch_pid.time_history,
                    'name': self.pitch_pid.name
                },
                'yaw_pid': {
                    'error': self.yaw_pid.error_history,
                    'setpoint': self.yaw_pid.setpoint_history,
                    'output': self.yaw_pid.output_history,
                    'time': self.yaw_pid.time_history,
                    'name': self.yaw_pid.name
                }
            }
            
            # Save the data to a file using pickle
            with open('flight_data.pkl', 'wb') as f:
                pickle.dump(flight_data, f)
            
            self.get_logger().info("Data saved to flight_data.pkl. Run plot_flight_data.py to visualize.")
        
    def gps_callback(self, msg):
        self.x = msg.point.x
        self.y = msg.point.y
        self.z = msg.point.z
        
        current_time = time.time() - self.start_time
        self.position_history['x'].append(self.x)
        self.position_history['y'].append(self.y)
        self.position_history['z'].append(self.z)
        self.position_history['time'].append(current_time)
        
        self.last_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1000000000.
        
    def imu_callback(self, msg):
        q = [
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z
        ]
        self.roll, self.pitch, self.yaw = self.quaternion_to_euler(q)
        
        current_time = time.time() - self.start_time
        self.attitude_history['roll'].append(self.roll)
        self.attitude_history['pitch'].append(self.pitch)
        self.attitude_history['yaw'].append(self.yaw)
        self.attitude_history['time'].append(current_time)
        
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
        if self.shutdown_requested:
            return
            
        if(time.time() - self.last_time < 1):
            current_time = time.time() - self.start_time
            
            # Create a more complex trajectory using multiple periodic functions
            # This will create a figure-8 pattern with varying height
            self.angle += 0.001
            
            # Figure-8 pattern in X-Y plane
            target_x = np.sin(self.angle) * 1.5
            target_y = np.sin(self.angle * 2) * 0.8
            
            # Add some variation in height using a different frequency
            target_z = 1.0 + 0.3 * np.sin(self.angle * 0.5)
            self.height_pid.setpoint = target_z
            
            # Add some acceleration and deceleration
            speed_factor = 0.5 + 0.5 * np.sin(self.angle * 0.25)
            self.angle += 0.0005 * speed_factor  # Variable speed
            
            # Target position with offset
            self.x_pos_pid.setpoint = target_x + 1
            self.y_pos_pid.setpoint = target_y + 1
            
            # Target yaw can point to a moving reference point
            reference_x = np.cos(self.angle * 1.3) * 0.5 + 1
            reference_y = np.sin(self.angle * 1.7) * 0.5 + 1
            target_yaw = np.arctan2(reference_y - self.y, reference_x - self.x)
            
            # Log target positions
            self.target_history['x'].append(target_x + 1)
            self.target_history['y'].append(target_y + 1)
            self.target_history['z'].append(target_z)
            self.target_history['yaw'].append(target_yaw)
            self.target_history['time'].append(current_time)
            
            print(f"Targets: X={target_x+1:.2f} Y={target_y+1:.2f} Z={target_z:.2f} Yaw={target_yaw:.2f}")
        
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
    
    def signal_handler(sig, frame):
        print("\nShutting down and saving flight data...")
        node.shutdown_requested = True
        node.save_data()
        # Allow some time for save_data to complete
        time.sleep(0.5)
        rclpy.shutdown()
        sys.exit(0)
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Make sure data is saved
        node.save_data()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()