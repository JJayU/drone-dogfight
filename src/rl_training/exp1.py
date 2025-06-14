from stable_baselines3 import PPO
from sim import CrazyflieEnv
import time
import math
import numpy as np
import os

class RLExperiment:
    def __init__(self, model_path):
        self.env = CrazyflieEnv()
        self.model = PPO.load(model_path, env=self.env)
        
        # Experiment parameters (now step-based for 200Hz simulation) render_mode="human"
        self.experiment_steps = 2100  # 6000 steps (30 seconds at 200Hz)
        self.hit_duration_steps = 140  # 400 steps to hold position (2 seconds at 200Hz)
        self.hit_threshold = 0.3      # 0.3m distance threshold
        self.yaw_threshold = 0.1      # ~5.7 degrees tolerance for yaw
        self.simulation_frequency = 70.0  # Hz
        
        # 16 static targets with orientation (x, y, z, yaw)
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
        
        # Home position
        self.home_position = [0.0, 0.0, 1.0, 0.0]
        
        # Experiment state
        self.current_target_index = 0
        self.current_target = None
        self.target_hit_start_step = None
        self.is_hitting_target = False
        self.targets_hit = 0
        self.total_targets = 0
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
        filename = f"rl_exp3_data_{self.exp_no}.txt"
        self.data_file_path = os.path.join(self.data_dir, filename)
        self.data_file = open(self.data_file_path, 'w')
        
        header = "time_seconds,targets_hit,total_targets,drone_x,drone_y,drone_z,drone_roll,drone_pitch,drone_yaw,target_x,target_y,target_z,target_yaw,distance_to_target,yaw_error,is_hitting\n"
        self.data_file.write(header)
        print(f"Started logging to: {self.data_file_path}")

    def log_data(self, step, obs):
        """Log current data to file"""
        if self.data_file and self.current_target:
            time_seconds = step / self.simulation_frequency  # Convert steps to seconds
            drone_pos = obs[0:3]
            drone_rpy = obs[6:9]
            distance = self.calculate_distance_to_target(drone_pos)
            yaw_error = self.calculate_yaw_error(drone_rpy[2])
            is_hitting = 1 if self.is_hitting_target else 0
            
            data_line = f"{time_seconds:.3f},{self.targets_hit},{self.total_targets}," \
                       f"{drone_pos[0]:.3f},{drone_pos[1]:.3f},{drone_pos[2]:.3f}," \
                       f"{drone_rpy[0]:.3f},{drone_rpy[1]:.3f},{drone_rpy[2]:.3f}," \
                       f"{self.current_target[0]:.3f},{self.current_target[1]:.3f},{self.current_target[2]:.3f}," \
                       f"{self.current_target[3]:.3f},{distance:.3f},{yaw_error:.3f},{is_hitting}\n"
            
            self.data_file.write(data_line)

    def stop_data_logging(self):
        """Stop logging and close file"""
        if self.data_file:
            self.data_file.close()
            self.data_file = None
            print(f"Data saved to: {self.data_file_path}")

    def next_target(self):
        """Get next target from the list"""
        if self.current_target_index >= len(self.defined_targets):
            self.current_target_index = 0
        target = self.defined_targets[self.current_target_index]
        self.current_target_index += 1
        self.total_targets += 1
        print(f"New target: {target} (Total targets attempted: {self.total_targets})")
        return target

    def calculate_distance_to_target(self, drone_pos):
        """Calculate 3D distance to current target"""
        if self.current_target is None:
            return float('inf')
        dx = drone_pos[0] - self.current_target[0]
        dy = drone_pos[1] - self.current_target[1]
        dz = drone_pos[2] - self.current_target[2]
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def calculate_yaw_error(self, drone_yaw):
        """Calculate yaw error to current target"""
        if self.current_target is None:
            return float('inf')
        return abs(self._wrap_angle(self.current_target[3] - drone_yaw))

    def check_target_hit(self, drone_pos, drone_yaw):
        """Check if target is hit and held for required duration"""
        distance = self.calculate_distance_to_target(drone_pos)
        yaw_error = self.calculate_yaw_error(drone_yaw)
        
        # Check if drone is close enough in position and orientation
        position_ok = distance <= self.hit_threshold
        orientation_ok = yaw_error <= self.yaw_threshold
        
        if position_ok and orientation_ok:
            if not self.is_hitting_target:
                self.is_hitting_target = True
                self.target_hit_start_step = self.current_step
                print(f"Targeting... (distance: {distance:.2f}m, yaw_error: {yaw_error:.2f}rad)")
            elif self.current_step - self.target_hit_start_step >= self.hit_duration_steps:
                self.targets_hit += 1
                print(f"TARGET HIT! #{self.targets_hit} - Moving to next target")
                self.current_target = self.next_target()
                self.is_hitting_target = False
                # Update environment target
                self.env.target_position = np.array(self.current_target[0:3])
                self.env.target_yaw = self.current_target[3]
        else:
            if self.is_hitting_target:
                print(f"Lost target (distance: {distance:.2f}m, yaw_error: {yaw_error:.2f}rad)")
                self.is_hitting_target = False

    def _wrap_angle(self, angle):
        """Normalize angle to range [-pi, pi]"""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def run_experiment(self):
        """Run the complete experiment"""
        print("\n" + "="*60)
        print("RL Model Experiment - Static Target Challenge (200Hz)")
        print("="*60)
        experiment_duration = self.experiment_steps / self.simulation_frequency
        hold_duration = self.hit_duration_steps / self.simulation_frequency
        print(f"Duration: {experiment_duration:.1f} seconds ({self.experiment_steps} steps at {self.simulation_frequency}Hz)")
        print(f"Total predefined targets: {len(self.defined_targets)}")
        print(f"Hit threshold: {self.hit_threshold}m")
        print(f"Hold duration: {hold_duration:.1f}s ({self.hit_duration_steps} steps)")
        print("="*60)
        print("Press Enter to start...")
        input()
        
        # Initialize experiment
        self.exp_no += 1
        self.targets_hit = 0
        self.total_targets = 0
        self.current_target_index = 0
        self.current_target = self.next_target()
        self.is_hitting_target = False
        self.current_step = 0
        self.start_data_logging()
        
        # Reset environment and set first target
        obs, _ = self.env.reset()
        self.env.target_position = np.array(self.current_target[0:3])
        self.env.target_yaw = self.current_target[3]
        
        print(f"\nExperiment {self.exp_no} started!")
        print(f"First target: {self.current_target}")
        
        last_status_step = 0
        
        while self.current_step < self.experiment_steps:
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Extract drone state (fixed index for yaw)
            drone_pos = obs[0:3]
            drone_yaw = obs[8]  # yaw is at index 8 in RPY according to observation
            
            # Check target hits
            self.check_target_hit(drone_pos, drone_yaw)
            
            # Log data every few steps to avoid too much I/O
            if self.current_step % 10 == 0:  # Log every 10 steps (every 0.05s at 200Hz)
                self.log_data(self.current_step, obs)
            
            # Reset if terminated/truncated
            if terminated or truncated:
                elapsed_time = self.current_step / self.simulation_frequency
                print(f"Environment reset at {elapsed_time:.1f}s")
                obs, _ = self.env.reset()
                # Restore current target after reset
                self.env.target_position = np.array(self.current_target[0:3])
                self.env.target_yaw = self.current_target[3]
            
            # Print status every 400 steps (every 2 seconds at 200Hz)
            if self.current_step - last_status_step >= 400:
                distance = self.calculate_distance_to_target(drone_pos)
                yaw_error = self.calculate_yaw_error(drone_yaw)
                remaining_steps = self.experiment_steps - self.current_step
                elapsed_time = self.current_step / self.simulation_frequency
                remaining_time = remaining_steps / self.simulation_frequency
                
                print(f"Time: {elapsed_time:.1f}s (remaining: {remaining_time:.1f}s) | "
                      f"Pos: [{drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f}] | "
                      f"Yaw: {drone_yaw:.2f} | Target: {self.current_target_index}/{len(self.defined_targets)} | "
                      f"Dist: {distance:.2f}m | Yaw_err: {yaw_error:.2f}rad | Hits: {self.targets_hit}")
                
                last_status_step = self.current_step
            
            self.current_step += 1
        
        # Experiment finished
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETE!")
        print("="*60)
        total_time = self.current_step / self.simulation_frequency
        print(f"Total time: {total_time:.1f} seconds ({self.current_step} steps)")
        print(f"Targets hit: {self.targets_hit}")
        print(f"Total targets attempted: {self.total_targets}")
        print(f"Available targets: {len(self.defined_targets)}")
        
        if self.total_targets > 0:
            success_rate = (self.targets_hit / self.total_targets) * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        targets_per_second = self.targets_hit / total_time if total_time > 0 else 0
        print(f"Targets per second: {targets_per_second:.2f}")
        print("="*60)
        
        self.stop_data_logging()
        
        # Return to home position
        print("Returning to home position...")
        self.return_home()

    def return_home(self):
        """Return drone to home position"""
        self.env.target_position = np.array(self.home_position[0:3])
        self.env.target_yaw = self.home_position[3]
        
        obs, _ = self.env.reset()  # Reset to ensure clean state
        
        # Run for a few steps to reach home (1000 steps = 5 seconds at 200Hz)
        home_steps = 0
        print("Flying home...")
        while home_steps < 1000:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if terminated or truncated:
                obs, _ = self.env.reset()
                self.env.target_position = np.array(self.home_position[0:3])
                self.env.target_yaw = self.home_position[3]
            
            # Check if close to home
            drone_pos = obs[0:3]
            distance_to_home = np.linalg.norm(drone_pos - np.array(self.home_position[0:3]))
            if distance_to_home < 0.2:
                print(f"Reached home position (distance: {distance_to_home:.2f}m)")
                break
                
            home_steps += 1
        
        print("Returned to home position")

    def print_targets_info(self):
        """Print information about all targets"""
        print("\nPredefined targets:")
        print("Index | X     | Y     | Z     | Yaw   | Distance from origin")
        print("-" * 60)
        for i, target in enumerate(self.defined_targets):
            x, y, z, yaw = target
            dist_from_origin = math.sqrt(x**2 + y**2 + z**2)
            print(f"{i+1:5d} | {x:5.1f} | {y:5.1f} | {z:5.1f} | {yaw:5.2f} | {dist_from_origin:5.2f}")
        print("-" * 60)

    def close(self):
        """Clean up resources"""
        self.stop_data_logging()
        self.env.close()


def main():
    # Configuration
    models_dir = "models/PPO"
    model_path = f"{models_dir}/zmiejszeniev2"  # Path to the model file (without .zip)
    
    # Check if model exists
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model not found at {model_path}.zip")
        print("Please check the model path and ensure the model file exists.")
        return
    
    # Create and run experiment
    experiment = RLExperiment(model_path)
    
    try:
        # Show targets information
        experiment.print_targets_info()
        
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