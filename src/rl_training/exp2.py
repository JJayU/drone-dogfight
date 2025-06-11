from stable_baselines3 import PPO
from sim import CrazyflieEnv
import time
import math
import numpy as np
import os

class RLExperiment5:
    def __init__(self, model_path):
        self.env = CrazyflieEnv()
        self.model = PPO.load(model_path, env=self.env)
        
        # Experiment parameters (step-based for 200Hz simulation)render_mode="human"
        self.experiment_steps = 12000  # 12000 steps (60 seconds at 200Hz)
        
        # Target trajectory parameters
        self.target_center_x = 0.0 
        self.target_center_y = 0.0  
        self.target_base_height = 2.0
        self.target_speed = 0.5
        self.figure8_radius = 3.0
        self.height_amplitude = 1.0
        
        # Following parameters
        self.follow_distance = 0.5
        self.follow_height_offset = 0.0
        
        # Current target state
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.target_direction_angle = 0.0
        
        # Experiment state
        self.start_step = 0
        self.current_step = 0
        
        # Data logging
        self.data_dir = "/home/ws/exp_data"
        self.data_file = None
        self.exp_no = 0
        self.setup_data_logging()

    def setup_data_logging(self):
        """Setup data logging directory"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created data directory: {self.data_dir}")

    def start_data_logging(self):
        """Start logging data to file"""
        filename = f"rl_exp5_data_{self.exp_no}.txt"
        self.data_file_path = os.path.join(self.data_dir, filename)
        self.data_file = open(self.data_file_path, 'w')
        
        header = "time,drone_x,drone_y,drone_z,drone_roll,drone_pitch,drone_yaw," \
                "target_x,target_y,target_z,target_direction,distance_to_target," \
                "aiming_error,desired_x,desired_y,desired_z,desired_yaw\n"
        self.data_file.write(header)
        print(f"Started logging to: {self.data_file_path}")

    def log_data(self, step, obs, desired_pos):
        """Log current data to file"""
        if self.data_file:
            time_elapsed = step / 200.0  # Convert steps to seconds
            drone_pos = obs[0:3]
            drone_rpy = obs[6:9]
            
            # Calculate distance to target
            distance_to_target = math.sqrt(
                (drone_pos[0] - self.target_x)**2 + 
                (drone_pos[1] - self.target_y)**2 + 
                (drone_pos[2] - self.target_z)**2
            )
            
            # Calculate aiming error
            target_bearing = math.atan2(self.target_y - drone_pos[1], self.target_x - drone_pos[0])
            aiming_error = abs(self._wrap_angle(drone_rpy[2] - target_bearing))
            
            data_line = f"{time_elapsed:.3f}," \
                       f"{drone_pos[0]:.3f},{drone_pos[1]:.3f},{drone_pos[2]:.3f}," \
                       f"{drone_rpy[0]:.3f},{drone_rpy[1]:.3f},{drone_rpy[2]:.3f}," \
                       f"{self.target_x:.3f},{self.target_y:.3f},{self.target_z:.3f}," \
                       f"{self.target_direction_angle:.3f},{distance_to_target:.3f}," \
                       f"{aiming_error:.3f}," \
                       f"{desired_pos[0]:.3f},{desired_pos[1]:.3f},{desired_pos[2]:.3f},{desired_pos[3]:.3f}\n"
            
            self.data_file.write(data_line)

    def stop_data_logging(self):
        """Stop logging and close file"""
        if self.data_file:
            self.data_file.close()
            self.data_file = None
            print(f"Data saved to: {self.data_file_path}")

    def calculate_target_position(self, elapsed_time):
        """Calculate target position on figure-8 trajectory with direction"""
        t = (elapsed_time * self.target_speed) / self.figure8_radius
        scale = self.figure8_radius
        
        # Figure-8 trajectory
        target_x = self.target_center_x + scale * np.sin(t)
        target_y = self.target_center_y + scale * np.sin(2 * t) / 2
        
        # Height variation (vertical wave)
        height_variation = self.height_amplitude * np.sin(0.5 * t)
        target_z = self.target_base_height + height_variation

        # Calculate direction angle
        dt = 0.01
        t_next = t + dt
        next_x = self.target_center_x + scale * np.sin(t_next)
        next_y = self.target_center_y + scale * np.sin(2 * t_next) / 2
        
        direction_angle = np.arctan2(next_y - target_y, next_x - target_x)

        return target_x, target_y, target_z, direction_angle

    def calculate_drone_desired_position(self, drone_pos):
        """Calculate desired drone position behind the target, aiming at it"""
        # Position behind the target
        rear_angle = self.target_direction_angle + np.pi
        desired_x = self.target_x + self.follow_distance * np.cos(rear_angle)
        desired_y = self.target_y + self.follow_distance * np.sin(rear_angle)
        desired_z = self.target_z + self.follow_height_offset
        
        # Yaw to aim at the target
        aim_yaw = np.arctan2(self.target_y - drone_pos[1], self.target_x - drone_pos[0])
        
        return desired_x, desired_y, desired_z, aim_yaw

    def _wrap_angle(self, angle):
        """Normalize angle to range [-pi, pi]"""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def run_experiment(self):
        """Run the complete experiment"""
        print("\n" + "="*60)
        print("RL Model Experiment 5 - Target Following (200Hz)")
        print("="*60)
        print("Cel porusza się po trajektorii ósemki z pionową falą")
        print("Dron śledzi cel i utrzymuje pozycję za nim, celując w jego tył")
        print(f"Duration: {self.experiment_steps} steps ({self.experiment_steps/200:.1f} seconds at 200Hz)")
        print("="*60)
        print("Naciśnij Enter, aby rozpocząć eksperyment...")
        input()
        
        # Initialize experiment
        self.exp_no += 1
        self.current_step = 0
        self.start_data_logging()
        
        # Reset environment
        obs, _ = self.env.reset()
        
        print(f"\n[Start] Eksperyment {self.exp_no} rozpoczęty.")
        
        last_status_step = 0
        
        while self.current_step < self.experiment_steps:
            # Calculate current time
            elapsed_time = self.current_step / 200.0
            
            # Update target position
            self.target_x, self.target_y, self.target_z, self.target_direction_angle = \
                self.calculate_target_position(elapsed_time)
            
            # Calculate desired drone position
            drone_pos = obs[0:3]
            desired_x, desired_y, desired_z, desired_yaw = \
                self.calculate_drone_desired_position(drone_pos)
            
            # Update environment target to desired position
            self.env.target_position = np.array([desired_x, desired_y, desired_z])
            self.env.target_yaw = desired_yaw
            
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Log data every 10 steps (every 0.05s at 200Hz)
            if self.current_step % 10 == 0:
                desired_pos = [desired_x, desired_y, desired_z, desired_yaw]
                self.log_data(self.current_step, obs, desired_pos)
            
            # Reset if terminated/truncated
            if terminated or truncated:
                print(f"Environment reset at step {self.current_step}")
                obs, _ = self.env.reset()
            
            # Print status every 400 steps (every 2 seconds at 200Hz)
            if self.current_step - last_status_step >= 400:
                distance_to_target = math.sqrt(
                    (drone_pos[0] - self.target_x)**2 + 
                    (drone_pos[1] - self.target_y)**2 + 
                    (drone_pos[2] - self.target_z)**2
                )
                
                target_bearing = math.atan2(self.target_y - drone_pos[1], self.target_x - drone_pos[0])
                drone_yaw = obs[8]  # yaw is at index 8
                aiming_error = abs(self._wrap_angle(drone_yaw - target_bearing))
                
                remaining_steps = self.experiment_steps - self.current_step
                remaining_time = remaining_steps / 200.0
                
                print(f"Time: {elapsed_time:.1f}s (remaining: {remaining_time:.1f}s) | "
                      f"Drone: [{drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f}] | "
                      f"Target: [{self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f}] | "
                      f"Distance: {distance_to_target:.2f}m | "
                      f"Aiming error: {aiming_error:.2f}rad")
                
                last_status_step = self.current_step
            
            self.current_step += 1
        
        # Experiment finished
        print("\n" + "="*60)
        print("[Koniec] Eksperyment zakończony!")
        print("="*60)
        print(f"Total time: {self.current_step/200.0:.1f} seconds")
        print("Cel poruszał się po ósemce, dron śledził go z tyłu.")
        print("="*60)
        
        self.stop_data_logging()
        
        # Return to home position
        print("Powrót do pozycji domowej...")
        self.return_home()

    def return_home(self):
        """Return drone to home position"""
        home_position = [0.0, 0.0, 2.0]
        home_yaw = 0.0
        
        self.env.target_position = np.array(home_position)
        self.env.target_yaw = home_yaw
        
        obs, _ = self.env.reset()  # Reset to ensure clean state
        
        # Run for a few steps to reach home (1000 steps = 5 seconds at 200Hz)
        home_steps = 0
        print("Powrót do domu...")
        while home_steps < 1000:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if terminated or truncated:
                obs, _ = self.env.reset()
                self.env.target_position = np.array(home_position)
                self.env.target_yaw = home_yaw
            
            # Check if close to home
            drone_pos = obs[0:3]
            distance_to_home = np.linalg.norm(drone_pos - np.array(home_position))
            if distance_to_home < 0.2:
                print(f"Reached home position (distance: {distance_to_home:.2f}m)")
                break
                
            home_steps += 1
        
        print("Powrócono do pozycji domowej")

    def close(self):
        """Clean up resources"""
        self.stop_data_logging()
        self.env.close()


def main():
    # Configuration
    models_dir = "models/PPO"
    model_path = f"{models_dir}/1100000"  # Path to the model file without .zip extension
    
    # Check if model exists
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model not found at {model_path}.zip")
        print("Please check the model path and ensure the model file exists.")
        return
    
    # Create and run experiment
    experiment = RLExperiment5(model_path)
    
    try:
        experiment.run_experiment()
        
        # Ask if user wants to run another experiment
        while True:
            response = input("\nRun another experiment? (y/n): ").lower().strip()
            if response == 'y':
                experiment.run_experiment()
            elif response == 'n':
                break
            else:
                print("Please enter 'y' or 'n'")
                
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
    finally:
        experiment.close()


if __name__ == "__main__":
    main()