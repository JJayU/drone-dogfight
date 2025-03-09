import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Imu
from std_srvs.srv import Empty
import numpy as np
import optuna
import threading
import time
import os
import matplotlib.pyplot as plt
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

class PIDTuner(Node):
    def __init__(self):
        super().__init__('pid_tuner')
        
        # Callback groups to avoid blocking
        self.timer_callback_group = MutuallyExclusiveCallbackGroup()
        self.service_callback_group = MutuallyExclusiveCallbackGroup()
        
        # Current drone state
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.last_time = 0.0
        
        # Position and orientation history for visualization
        self.history = {
            'time': [],
            'x': [], 'y': [], 'z': [],
            'roll': [], 'pitch': [], 'yaw': [],
            'target_x': [], 'target_y': [], 'target_z': []
        }
        
        # Subscriptions
        self.gps_sub = self.create_subscription(PointStamped, 'gps', self.gps_callback, 10)
        self.imu_sub = self.create_subscription(Imu, 'imu', self.imu_callback, 10)
        
        # Publisher
        self.motor_pub = self.create_publisher(Float32MultiArray, 'motor_power', 10)
        
        # Reset service client
        self.reset_client = self.create_client(
            Empty, 'reset_simulation', callback_group=self.service_callback_group)
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Reset service not available, waiting...')
        
        # Optimization variables
        self.current_trial = None
        self.max_iterations = 1000  # Increased from 2000 for better results
        self.error_samples = {}
        self.optimization_complete = threading.Event()
        
        # Control timer
        self.control_timer = self.create_timer(
            0.005, self.control_callback, callback_group=self.timer_callback_group)
        self.control_timer.cancel()
        
        # Start optimization in separate thread
        self.optimization_thread = threading.Thread(target=self.run_optimization)
        self.optimization_thread.start()
        
        # Create plots directory
        os.makedirs('/home/ws/plots', exist_ok=True)
        
    def run_optimization(self):
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(self.objective, n_trials=500)  # Reduced for development, increase for production
        
        self.best_params = self.study.best_params
        self.get_logger().info(f"Best parameters: {self.best_params}")
        
        # Run one final time with best parameters and save plots
        self.visualize_best_params()
        
    def visualize_best_params(self):
        """Run a final trial with best parameters and generate plots"""
        reset_future = self.reset_simulation()
        while not reset_future.done():
            time.sleep(0.01)
        
        time.sleep(0.5)
        
        # Clear history and ensure synchronization
        self.history_lock = threading.Lock()
        for key in self.history:
            self.history[key] = []
            
        # Reset error samples
        self.error_samples = {
            "x": [], "y": [], "z": [],
            "roll": [], "pitch": [], "yaw": []
        }
        
        # Set targets
        self.target_x = 1.0
        self.target_y = 1.0
        self.target_z = 1.0
        self.target_yaw = 0.7
        
        # Initialize PIDs with best parameters
        d_filter = self.best_params.get('d_filter', 0.8)
        
        self.x_pos_pid = PID(self.best_params['kp_x'], self.best_params['ki_x'], 
                           self.best_params['kd_x'], setpoint=self.target_x, name="X Position", d_filter=d_filter)
        self.y_pos_pid = PID(self.best_params['kp_y'], self.best_params['ki_y'], 
                           self.best_params['kd_y'], setpoint=self.target_y, name="Y Position", d_filter=d_filter)
        self.height_pid = PID(self.best_params['kp_z'], self.best_params['ki_z'], 
                           self.best_params['kd_z'], setpoint=self.target_z, name="Height", d_filter=d_filter)
        
        self.roll_pid = PID(self.best_params['kp_roll'], self.best_params['ki_roll'], 
                          self.best_params['kd_roll'], setpoint=0.0, name="Roll", d_filter=d_filter)
        self.pitch_pid = PID(self.best_params['kp_pitch'], self.best_params['ki_pitch'], 
                           self.best_params['kd_pitch'], setpoint=0.0, name="Pitch", d_filter=d_filter)
        self.yaw_pid = PID(self.best_params['kp_yaw'], self.best_params['ki_yaw'], 
                         self.best_params['kd_yaw'], setpoint=self.target_yaw, name="yaw", d_filter=d_filter)
        
        # Run visualization trial
        self.timer_counter = 0
        self.recording_data = True
        self.data_complete = threading.Event()
        self.control_timer = self.create_timer(
            0.005, self.control_callback, callback_group=self.timer_callback_group)
        
        # Wait for completion
        end_time = time.time() + self.max_iterations * 0.005 + 1.0
        while time.time() < end_time:
            if self.timer_counter >= self.max_iterations:
                break
            time.sleep(0.1)
            
        self.control_timer.cancel()
        self.recording_data = False
        time.sleep(0.2)  # Allow final callbacks to complete
        
        # Generate and save plots
        self.generate_plots()
        
    def generate_plots(self):
        # Create plots directory
        plots_dir = '/home/ws/plots'
        os.makedirs(plots_dir, exist_ok=True)
        
        # Ensure all arrays have the same length
        min_length = min(len(self.history['time']), 
                         len(self.history['x']), 
                         len(self.history['y']), 
                         len(self.history['z']))
        
        # Truncate all arrays to the same length
        time_values = np.array(self.history['time'][:min_length])
        x_values = np.array(self.history['x'][:min_length])
        y_values = np.array(self.history['y'][:min_length])
        z_values = np.array(self.history['z'][:min_length])
        
        # Ensure orientation arrays have the same length
        min_orient_length = min(min_length, 
                              len(self.history['roll']), 
                              len(self.history['pitch']), 
                              len(self.history['yaw']))
        roll_values = np.array(self.history['roll'][:min_orient_length])
        pitch_values = np.array(self.history['pitch'][:min_orient_length])
        yaw_values = np.array(self.history['yaw'][:min_orient_length])
        time_for_orient = time_values[:min_orient_length]
        
        # Error arrays should already be synchronized from control_callback
        errors_x = np.array(self.error_samples['x'])
        errors_y = np.array(self.error_samples['y'])
        errors_z = np.array(self.error_samples['z'])
        error_time = np.arange(len(errors_x)) * 0.005
        
        self.get_logger().info(f"Plotting data: time={len(time_values)}, x={len(x_values)}, y={len(y_values)}, z={len(z_values)}")
        
        # Normalize time relative to start
        if len(time_values) > 0:
            time_values = time_values - time_values[0]
            if len(time_for_orient) > 0:
                time_for_orient = time_for_orient - time_for_orient[0]
        
        # Position trajectory plot
        plt.figure(figsize=(12, 8))
        
        # 3D trajectory
        ax1 = plt.subplot(221, projection='3d')
        if min_length > 0:
            ax1.plot(x_values, y_values, z_values, 'b-', label='Actual')
            ax1.plot([self.target_x], [self.target_y], [self.target_z], 'ro', label='Target')
            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            ax1.set_zlabel('Z Position')
            ax1.set_title('3D Trajectory')
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 0.5, "Insufficient data", ha='center')
        
        # Position over time
        ax2 = plt.subplot(222)
        if min_length > 0:
            ax2.plot(time_values, x_values, 'r-', label='X')
            ax2.plot(time_values, y_values, 'g-', label='Y')
            ax2.plot(time_values, z_values, 'b-', label='Z')
            ax2.axhline(y=self.target_x, color='r', linestyle='--')
            ax2.axhline(y=self.target_y, color='g', linestyle='--')
            ax2.axhline(y=self.target_z, color='b', linestyle='--')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Position')
            ax2.set_title('Position over Time')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "Insufficient data", ha='center')
        
        # Orientation over time
        ax3 = plt.subplot(223)
        if min_orient_length > 0:
            ax3.plot(time_for_orient, np.rad2deg(roll_values), 'r-', label='Roll')
            ax3.plot(time_for_orient, np.rad2deg(pitch_values), 'g-', label='Pitch')
            ax3.plot(time_for_orient, np.rad2deg(yaw_values), 'b-', label='Yaw')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Angle (degrees)')
            ax3.set_title('Orientation over Time')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, "Insufficient orientation data", ha='center')
        
        # Errors over time
        ax4 = plt.subplot(224)
        if len(errors_x) > 0:
            ax4.plot(error_time, errors_x, 'r-', label='X error')
            ax4.plot(error_time, errors_y, 'g-', label='Y error')
            ax4.plot(error_time, errors_z, 'b-', label='Z error')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Error')
            ax4.set_title('Position Errors over Time')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, "No error data collected", ha='center')
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/trajectory_and_errors.png')
        
        # Save 2D trajectory (top view)
        plt.figure(figsize=(10, 8))
        if min_length > 0:
            plt.plot(x_values, y_values, 'b-', label='Drone Path')
            plt.plot(self.target_x, self.target_y, 'ro', label='Target Position')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title('Drone Trajectory (Top View)')
            plt.grid(True)
            plt.axis('equal')
            plt.legend()
        else:
            plt.text(0.5, 0.5, "Insufficient data for 2D trajectory", ha='center')
        plt.savefig(f'{plots_dir}/2d_trajectory.png')
                
        # Save error metrics
        mse_errors = {}
        for key, samples in self.error_samples.items():
            if samples:
                mse_errors[key] = np.mean(np.square(samples))
            else:
                mse_errors[key] = float('inf')
                
        with open(f'{plots_dir}/error_metrics.txt', 'w') as f:
            f.write("Mean Squared Errors:\n")
            for key, value in mse_errors.items():
                f.write(f"{key}: {value:.6f}\n")
            f.write("\nBest Parameters:\n")
            for key, value in self.best_params.items():
                f.write(f"{key}: {value:.6f}\n")
                
        self.get_logger().info(f"Plots and error metrics saved to {plots_dir}")
        
    def reset_simulation(self):
        request = Empty.Request()
        return self.reset_client.call_async(request)
        
    def gps_callback(self, msg):
        self.x = msg.point.x
        self.y = msg.point.y
        self.z = msg.point.z
        self.last_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1000000000.
        
        # Record history if in visualization mode
        if hasattr(self, 'recording_data') and self.recording_data:
            self.history['time'].append(self.last_time)
            self.history['x'].append(self.x)
            self.history['y'].append(self.y)
            self.history['z'].append(self.z)
            self.history['target_x'].append(self.target_x)
            self.history['target_y'].append(self.target_y)
            self.history['target_z'].append(self.target_z)
        
    def imu_callback(self, msg):
        q = [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]
        self.roll, self.pitch, self.yaw = self.quaternion_to_euler(q)
        
        # Record history if in visualization mode
        if hasattr(self, 'recording_data') and self.recording_data:
            self.history['roll'].append(self.roll)
            self.history['pitch'].append(self.pitch)
            self.history['yaw'].append(self.yaw)
        
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
    
    def objective(self, trial):
        self.current_trial = trial
        
        # Reset simulation
        reset_future = self.reset_simulation()
        while not reset_future.done():
            time.sleep(0.01)
        
        # Reset error data
        self.error_samples = {
            "x": [], "y": [], "z": [],
            "roll": [], "pitch": [], "yaw": []
        }
        
        # Target setpoints
        self.target_x = 1.0
        self.target_y = 1.0
        self.target_z = 1.0
        self.target_yaw = 0.0

        # PID parameters with expanded ranges
        kp_x = trial.suggest_float('kp_x', 0.05, 1.0)
        ki_x = trial.suggest_float('ki_x', 0.0, 0.2)
        kd_x = trial.suggest_float('kd_x', 0.01, 0.5)
        
        kp_y = trial.suggest_float('kp_y', 0.05, 1.0)
        ki_y = trial.suggest_float('ki_y', 0.0, 0.2)
        kd_y = trial.suggest_float('kd_y', 0.01, 0.5)
        
        kp_z = trial.suggest_float('kp_z', 0.3, 2.0)
        ki_z = trial.suggest_float('ki_z', 0.0, 0.7)
        kd_z = trial.suggest_float('kd_z', 0.1, 1.0)
        
        kp_roll = trial.suggest_float('kp_roll', 0.01, 0.3)
        ki_roll = trial.suggest_float('ki_roll', 0.0, 0.08)
        kd_roll = trial.suggest_float('kd_roll', 0.002, 0.05)
        
        kp_pitch = trial.suggest_float('kp_pitch', 0.01, 0.3)
        ki_pitch = trial.suggest_float('ki_pitch', 0.0, 0.08)
        kd_pitch = trial.suggest_float('kd_pitch', 0.002, 0.05)
        
        kp_yaw = trial.suggest_float('kp_yaw', 0.003, 0.05)
        ki_yaw = trial.suggest_float('ki_yaw', 0.0, 0.02)
        kd_yaw = trial.suggest_float('kd_yaw', 0.002, 0.05)
        
        d_filter = trial.suggest_float('d_filter', 0.5, 0.95)
        
        # Initialize PID controllers
        self.x_pos_pid = PID(kp_x, ki_x, kd_x, setpoint=self.target_x, name="X Position", d_filter=d_filter)
        self.y_pos_pid = PID(kp_y, ki_y, kd_y, setpoint=self.target_y, name="Y Position", d_filter=d_filter)
        self.height_pid = PID(kp_z, ki_z, kd_z, setpoint=self.target_z, name="Height", d_filter=d_filter)
        
        self.roll_pid = PID(kp_roll, ki_roll, kd_roll, setpoint=0.0, name="Roll", d_filter=d_filter)
        self.pitch_pid = PID(kp_pitch, ki_pitch, kd_pitch, setpoint=0.0, name="Pitch", d_filter=d_filter)
        self.yaw_pid = PID(kp_yaw, ki_yaw, kd_yaw, setpoint=self.target_yaw, name="yaw", d_filter=d_filter)
        
        # Reset timer counter
        self.timer_counter = 0
        self.optimization_complete.clear()
        
        # Start timer for this trial
        self.control_timer = self.create_timer(
            0.005, self.control_callback, callback_group=self.timer_callback_group)
        
        # Wait for trial completion
        self.optimization_complete.wait()
        
        # Calculate errors
        mse_errors = {}
        for key, samples in self.error_samples.items():
            if samples:
                mse_errors[key] = np.mean(np.square(samples))
            else:
                mse_errors[key] = float('inf')
        
        # Weights for errors
        weights = {
            "x": 1.0, "y": 1.0, "z": 1.0,
            "roll": 0.5, "pitch": 0.5, "yaw": 0.2,
        }
        
        # Total weighted error
        total_error = sum(mse_errors[key] * weights[key] for key in mse_errors)
        
        self.get_logger().info(f"Trial {trial.number} finished with error: {total_error:.6f}")
        return total_error
    
    def control_callback(self):
        self.timer_counter += 1
        
        if (time.time() - self.last_time < 1):
            # Calculate desired roll and pitch based on position
            desired_pitch = self.x_pos_pid.update(self.x, 0.005)
            desired_roll = self.y_pos_pid.update(self.y, 0.005)
            
            # Limit angles
            desired_pitch = np.clip(desired_pitch, -0.3, 0.3)
            desired_roll = np.clip(desired_roll, -0.3, 0.3)
            
            # Transform roll and pitch considering current yaw
            self.pitch_pid.setpoint = desired_pitch * np.cos(self.yaw) - desired_roll * np.sin(-self.yaw)
            self.roll_pid.setpoint = -desired_roll * np.cos(-self.yaw) + desired_pitch * np.sin(self.yaw)
            
            # Calculate controls
            height_control = self.height_pid.update(self.z, 0.005)
            pitch_control = self.pitch_pid.update(self.pitch, 0.005)
            roll_control = self.roll_pid.update(self.roll, 0.005)
            yaw_control = self.yaw_pid.update(self.yaw, 0.005)
            
            # Collect error samples
            self.error_samples["x"].append(abs(self.x - self.target_x))
            self.error_samples["y"].append(abs(self.y - self.target_y))
            self.error_samples["z"].append(abs(self.z - self.target_z))
            self.error_samples["roll"].append(abs(self.roll - self.roll_pid.setpoint))
            self.error_samples["pitch"].append(abs(self.pitch - self.pitch_pid.setpoint))
            self.error_samples["yaw"].append(abs(self.yaw - self.target_yaw))
            
            # Calculate motor commands
            m1 = height_control - pitch_control + roll_control - yaw_control
            m2 = height_control - pitch_control - roll_control + yaw_control
            m3 = height_control + pitch_control - roll_control - yaw_control
            m4 = height_control + pitch_control + roll_control + yaw_control
            
            # Scale motor powers
            max_thrust = max(abs(m1), abs(m2), abs(m3), abs(m4))
            if max_thrust > 1.0:
                m1 /= max_thrust
                m2 /= max_thrust
                m3 /= max_thrust
                m4 /= max_thrust
            
            # Publish motor commands
            motor_commands = Float32MultiArray()
            motor_commands.data = [float(m1), float(m2), float(m3), float(m4)]
            self.motor_pub.publish(motor_commands)
        
        # Check if trial is complete
        if self.timer_counter >= self.max_iterations:
            self.control_timer.cancel()
            self.optimization_complete.set()

class PID:
    def __init__(self, kp, ki, kd, setpoint, name="unnamed", d_filter=0.8):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0
        self.name = name
        self.prev_derivative = 0
        self.d_filter = d_filter
        
    def update(self, measured_value, dt):
        if self.name == "yaw":  
            error = np.arctan2(np.sin(self.setpoint - measured_value), np.cos(self.setpoint - measured_value))
        else:
            error = self.setpoint - measured_value
        
        self.integral = np.clip(self.integral + error * dt, -5, 5)

        derivative = (error - self.previous_error) / max(dt, 0.001)
        derivative = self.d_filter * self.prev_derivative + (1 - self.d_filter) * derivative
        self.prev_derivative = derivative
        
        p_term = self.kp * error
        i_term = self.ki * self.integral
        d_term = self.kd * derivative
        
        output = p_term + i_term + d_term
        self.previous_error = error
        
        return output

def main():
    rclpy.init()
    node = PIDTuner()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()