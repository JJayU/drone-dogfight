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
        
        # Experiment parameters (step-based for 50Hz simulation)render_mode="human"
        self.experiment_steps = 3000  # 3000 steps (60 seconds at 50Hz)
        
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
        filename = f"exp5_data_{self.exp_no}.txt"
        self.data_file_path = os.path.join(self.data_dir, filename)
        
        with open(f'/home/ws/exp_data/exp5_data_{self.exp_no}.txt', 'w') as f:
            f.write("time, drone_x, drone_y, drone_z, drone_roll, drone_pitch, drone_yaw, "
                    "target_x, target_y, target_z, target_direction, distance_to_target, "
                    "aiming_error, m1_power, m2_power, m3_power, m4_power, cumulative_energy\n")
    
        # Open file for continuous writing
        self.data_file = open(self.data_file_path, 'a')
        print(f"Started logging to: {self.data_file_path}")

    def log_data(self, step, obs, desired_pos, action):
        """Log current data to file including motor powers and cumulative energy"""
        if self.data_file:
            time_elapsed = step / 50.0  # Convert steps to seconds (50Hz)
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
            
            # Extract motor powers from environment
            if hasattr(self.env, 'data') and hasattr(self.env.data, 'ctrl'):
                motor_powers = self.env.data.ctrl[:4]# Get actual motor commands from MuJoCo
            else:
                # Fallback: use action if it represents motor powers
                motor_powers = action[:4] if len(action) >= 4 else [0.0, 0.0, 0.0, 0.0]
            
            # Simple cumulative energy - sum of all motor powers
            if not hasattr(self, 'cumulative_energy'):
                self.cumulative_energy = 0.0
            
            # Add current motor powers to cumulative sum
            current_total_power = sum(motor_powers)
            self.cumulative_energy += current_total_power
            
            data_line = f"{time_elapsed:.3f}, " \
                       f"{drone_pos[0]:.3f}, {drone_pos[1]:.3f}, {drone_pos[2]:.3f}, " \
                       f"{drone_rpy[0]:.3f}, {drone_rpy[1]:.3f}, {drone_rpy[2]:.3f}, " \
                       f"{self.target_x:.3f}, {self.target_y:.3f}, {self.target_z:.3f}, " \
                       f"{self.target_direction_angle:.3f}, {distance_to_target:.3f}, " \
                       f"{aiming_error:.3f}, " \
                       f"{motor_powers[0]:.3f}, {motor_powers[1]:.3f}, {motor_powers[2]:.3f}, {motor_powers[3]:.3f}, " \
                       f"{self.cumulative_energy:.3f}\n"
            
            self.data_file.write(data_line)
            self.data_file.flush()  # Ensure data is written immediately

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
        dt = 1/50  # Updated for 50Hz
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
        print("RL Model Experiment 5 - Target Following (50Hz)")
        print("="*60)
        print("Cel porusza się po trajektorii ósemki z pionową falą")
        print("Dron śledzi cel i utrzymuje pozycję za nim, celując w jego tył")
        print(f"Duration: {self.experiment_steps} steps ({self.experiment_steps/50:.1f} seconds at 50Hz)")
        print("="*60)
        print("Naciśnij Enter, aby rozpocząć eksperyment...")
        input()
        
        # Initialize experiment
        self.exp_no += 1
        self.current_step = 0
        self.cumulative_energy = 0.0  # Reset cumulative energy
        self.start_data_logging()
        
        # Reset environment
        obs, _ = self.env.reset()
        
        print(f"\n[Start] Eksperyment {self.exp_no} rozpoczęty.")
        
        last_status_step = 0
        
        while self.current_step < self.experiment_steps:
            # Calculate current time
            elapsed_time = self.current_step / 50  # Updated for 50Hz
            
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
            
            # Log data every 5 steps (every 0.1s at 50Hz)
            if self.current_step % 5 == 0:
                desired_pos = [desired_x, desired_y, desired_z, desired_yaw]
                self.log_data(self.current_step, obs, desired_pos, action)
            
            # Reset if terminated/truncated
            if terminated or truncated:
                print(f"Environment reset at step {self.current_step}")
                obs, _ = self.env.reset()
            
            # Print status every 100 steps (every 2 seconds at 50Hz)
            if self.current_step - last_status_step >= 100:
                distance_to_target = math.sqrt(
                    (drone_pos[0] - self.target_x)**2 + 
                    (drone_pos[1] - self.target_y)**2 + 
                    (drone_pos[2] - self.target_z)**2
                )
                
                target_bearing = math.atan2(self.target_y - drone_pos[1], self.target_x - drone_pos[0])
                drone_yaw = obs[8]  # yaw is at index 8
                aiming_error = abs(self._wrap_angle(drone_yaw - target_bearing))
                
                remaining_steps = self.experiment_steps - self.current_step
                remaining_time = remaining_steps / 50.0  # Updated for 50Hz
                
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
        print(f"Total time: {self.current_step/50.0:.1f} seconds")  # Updated for 50Hz
        print("Cel poruszał się po ósemce, dron śledził go z tyłu.")
        
        # Analyze power consumption
        power_stats = self.analyze_power_consumption()
        if power_stats:
            print(f"Total power consumption: {power_stats['total_energy']:.3f}")
            print(f"Average power per step: {power_stats['average_power_per_step']:.3f}")

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
        
        # Run for a few steps to reach home (250 steps = 5 seconds at 50Hz)
        home_steps = 0
        print("Powrót do domu...")
        while home_steps < 250:  # Updated for 50Hz
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

    def analyze_power_consumption(self):
        """Analyze power consumption from the experiment"""
        if hasattr(self, 'cumulative_energy'):
            total_time = self.current_step / 50.0  # Updated for 50Hz
            average_power = self.cumulative_energy / self.current_step if self.current_step > 0 else 0
            
            print(f"\nPower Consumption Analysis:")
            print(f"Total cumulative energy: {self.cumulative_energy:.3f}")
            print(f"Average power per step: {average_power:.3f}")
            print(f"Power consumption rate: {self.cumulative_energy / total_time:.3f} per second" if total_time > 0 else "N/A")
            
            return {
                'total_energy': self.cumulative_energy,
                'average_power_per_step': average_power,
                'total_time': total_time
            }
        return None


def main():
    # Configuration
    models_dir = "models/PPO"
    model_path = f"{models_dir}/zmiejszeniev2"  # Path to the model file without .zip extension
    
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