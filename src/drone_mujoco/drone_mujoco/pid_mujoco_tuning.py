import numpy as np
import optuna
import rclpy
from rclpy.node import Node
import mujoco
import mujoco_viewer
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Imu
import matplotlib.pyplot as plt

class CascadedPIDController:
    def __init__(self):
        # Outer loop (Position)
        self.pos_pid = {
            'x': {'kp': 0, 'ki': 0, 'kd': 0, 'integral': 0, 'prev_error': 0, 'prev_derivative': 0},
            'y': {'kp': 0, 'ki': 0, 'kd': 0, 'integral': 0, 'prev_error': 0, 'prev_derivative': 0}
        }
        
        # Inner loop (Angle)
        self.angle_pid = {
            'roll': {'kp': 0, 'ki': 0, 'kd': 0, 'integral': 0, 'prev_error': 0, 'prev_derivative': 0},
            'pitch': {'kp': 0, 'ki': 0, 'kd': 0, 'integral': 0, 'prev_error': 0, 'prev_derivative': 0}
        }
        
        # Height controller
        self.height_pid = {'kp': 0, 'ki': 0, 'kd': 0, 'integral': 0, 'prev_error': 0, 'prev_derivative': 0}
        
        # Yaw controller
        self.yaw_pid = {'kp': 0, 'ki': 0, 'kd': 0, 'integral': 0, 'prev_error': 0, 'prev_derivative': 0}

    def update_pid(self, pid_config, setpoint, measured_value, dt):
        """Generic PID update method"""
        error = setpoint - measured_value
        
        # Integral with anti-windup
        pid_config['integral'] = np.clip(pid_config['integral'] + error * dt, -0.3, 0.3)
        
        # Derivative with low-pass filter
        derivative = (error - pid_config['prev_error']) / max(dt, 0.001)
        derivative = 0.8 * pid_config['prev_derivative'] + 0.2 * derivative
        
        # PID terms
        p_term = pid_config['kp'] * error
        i_term = pid_config['ki'] * pid_config['integral']
        d_term = pid_config['kd'] * derivative
        
        # Update historical values
        pid_config['prev_error'] = error
        pid_config['prev_derivative'] = derivative
        
        return p_term + i_term + d_term

    def cascaded_position_control(self, current_pos, current_angle, setpoint_pos, dt):
        # Parametry konwersji
        pos_to_angle_kp = 0.1  
        pos_feedforward_kp = 0.2  # Dodatkowy współczynnik korekcji

        # Kontrola pozycji (zewnętrzna pętla)
        x_pos_control = self.update_pid(self.pos_pid['x'], setpoint_pos['x'], current_pos['x'], dt)
        y_pos_control = self.update_pid(self.pos_pid['y'], setpoint_pos['y'], current_pos['y'], dt)
        
        # Konwersja błędu pozycji na pożądane kąty z uwzględnieniem kontroli pozycji
        desired_roll = -(setpoint_pos['y'] - current_pos['y']) * pos_to_angle_kp + y_pos_control * pos_feedforward_kp
        desired_pitch = (setpoint_pos['x'] - current_pos['x']) * pos_to_angle_kp + x_pos_control * pos_feedforward_kp
        
        # Kontrola kątów (wewnętrzna pętla)
        roll_angle_control = self.update_pid(self.angle_pid['roll'], desired_roll, current_angle['roll'], dt)
        pitch_angle_control = self.update_pid(self.angle_pid['pitch'], desired_pitch, current_angle['pitch'], dt)
        
        return roll_angle_control, pitch_angle_control  

class MujocoPIDOptimizer:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.controller = CascadedPIDController()
        self.best_performance = float('inf')  # Inicjalizacja najlepszego wyniku
        self.best_trial_number = None  # Inicjalizacja numeru najlepszej próby
        
    def simulate_drone_response(self, pid_params, trial_number=None):
        # Recreate data to reset simulation
        self.data = mujoco.MjData(self.model)
        performance = float('inf')
        # Set PID parameters
        self.controller.height_pid['kp'] = pid_params.get('height_kp', 8.0)
        self.controller.height_pid['ki'] = pid_params.get('height_ki', 0.02)
        self.controller.height_pid['kd'] = pid_params.get('height_kd', 2.0)
        
        self.controller.angle_pid['roll']['kp'] = pid_params.get('roll_kp', 3.0)
        self.controller.angle_pid['roll']['ki'] = pid_params.get('roll_ki', 0.01)
        self.controller.angle_pid['roll']['kd'] = pid_params.get('roll_kd', 0.5)
        
        self.controller.angle_pid['pitch']['kp'] = pid_params.get('pitch_kp', 3.0)
        self.controller.angle_pid['pitch']['ki'] = pid_params.get('pitch_ki', 0.01)
        self.controller.angle_pid['pitch']['kd'] = pid_params.get('pitch_kd', 0.5)
        
        # Simulation parameters
        dt = 0.005
        total_time = 20.0
        setpoints = {
            'height': 1.0,
            'pos': {'x': 1.0, 'y': 1.0},
            'yaw': 0.0
        }
        
        # Performance tracking
        position_errors = []
        height_errors = []
        angle_errors = {
            'roll': [],
            'pitch': [],
            'yaw': []
        }
        stability_penalties = []
        
        time_steps = []
        for t in np.arange(0, total_time, dt):
            # Current state
            current_pos = {
                'x': self.data.qpos[0], 
                'y': self.data.qpos[1], 
                'z': self.data.qpos[2]
            }
            
            # Extract angles (assuming quaternion order)
            current_angles = self.quaternion_to_euler(self.data.qpos[3:7])
            current_angle = {
                'roll': current_angles[0],
                'pitch': current_angles[1],
                'yaw': current_angles[2]
            }
            
            # Height control
            height_control = self.controller.update_pid(
                self.controller.height_pid, 
                setpoints['height'], 
                current_pos['z'], 
                dt
            )
            
            # Cascaded position control
            roll_control, pitch_control = self.controller.cascaded_position_control(
                current_pos, 
                current_angle, 
                setpoints['pos'], 
                dt
            )
            
            # Yaw control
            yaw_control = self.controller.update_pid(
                self.controller.yaw_pid, 
                setpoints['yaw'], 
                current_angle['yaw'], 
                dt
            )
            
            # Compute motor commands
            m1 = 0.5 + height_control - pitch_control + roll_control - yaw_control
            m2 = 0.5 + height_control - pitch_control - roll_control + yaw_control
            m3 = 0.5 + height_control + pitch_control - roll_control - yaw_control
            m4 = 0.5 + height_control + pitch_control + roll_control + yaw_control
            
            # Set motor controls
            motor_commands = [m1, m2, m3, m4]
            for i in range(4):
                self.data.ctrl[i] = np.clip(motor_commands[i], 0, 1)
            
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            
           # Track errors
            position_errors.append(np.sqrt(
                (current_pos['x'] - setpoints['pos']['x'])**2 + 
                (current_pos['y'] - setpoints['pos']['y'])**2
            ))
            height_errors.append(abs(current_pos['z'] - setpoints['height']))
            
            # Track angle errors
            angle_errors['roll'].append(abs(current_angle['roll']))
            angle_errors['pitch'].append(abs(current_angle['pitch']))
            angle_errors['yaw'].append(abs(current_angle['yaw']))
            
            # Stability penalty (penalize large oscillations)
            stability_penalty = 0
            if len(angle_errors['roll']) > 10:
                roll_std = np.std(angle_errors['roll'][-10:])
                pitch_std = np.std(angle_errors['pitch'][-10:])
                yaw_std = np.std(angle_errors['yaw'][-10:])
                stability_penalty = roll_std + pitch_std + yaw_std
            stability_penalties.append(stability_penalty)
            
            time_steps.append(t)
        
        
        # Visualization
        if trial_number is not None:
            plt.figure(figsize=(12, 8))
            plt.subplot(3, 1, 1)
            plt.plot(time_steps, position_errors, label='Position Error')
            plt.xlabel('Time (s)')
            plt.ylabel('Error')
            plt.legend()

            plt.subplot(3, 1, 2)
            plt.plot(time_steps, height_errors, label='Height Error')
            plt.xlabel('Time (s)')
            plt.ylabel('Error')
            plt.legend()

            plt.subplot(3, 1, 3)
            plt.plot(time_steps, angle_errors['roll'], label='Roll Error')
            plt.plot(time_steps, angle_errors['pitch'], label='Pitch Error')
            plt.plot(time_steps, angle_errors['yaw'], label='Yaw Error')
            plt.xlabel('Time (s)')
            plt.ylabel('Error')
            plt.legend()

            plt.tight_layout()

            # Zapis wykresu tylko dla najlepszej próby
            if performance < self.best_performance:
                self.best_performance = performance
                self.best_trial_number = trial_number
                plot_filename = f'/home/ws/plots/trial_{trial_number}_performance_{performance:.2f}.png'  # Zmień ścieżkę, jeśli potrzebujesz
                plt.savefig(plot_filename)
            
            plt.close()  # Zamknij wykres, aby zwolnić pamięć
        # Compute performance metric
        mean_position_error = np.mean(position_errors)
        mean_height_error = np.mean(height_errors)
        mean_stability_penalty = np.mean(stability_penalties)
        
        # Combined performance metric
        performance = (
            mean_position_error * 10 +  # Emphasize position accuracy
            mean_height_error * 10 +    # Emphasize height precision
            mean_stability_penalty * 5  # Penalize instability
        )
        
        return performance
    
    def quaternion_to_euler(self, q):
        """Convert quaternion to Euler angles"""
        w, x, y, z = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def optimize_pid_parameters(self):
        def objective(trial):
            # Define parameter search space with cascaded control in mind
            pid_params = {
                'height_kp': trial.suggest_float('height_kp', 1, 20),
                'height_ki': trial.suggest_float('height_ki', 1, 20),
                'height_kd': trial.suggest_float('height_kd', 1, 20),
                
                'roll_kp': trial.suggest_float('roll_kp', 1, 20),
                'roll_ki': trial.suggest_float('roll_ki', 1, 20),
                'roll_kd': trial.suggest_float('roll_kd', 1, 20),
                
                'pitch_kp': trial.suggest_float('pitch_kp', 1, 20),
                'pitch_ki': trial.suggest_float('pitch_ki', 1, 20),
                'pitch_kd': trial.suggest_float('pitch_kd', 1, 20)
            }
            
            # Run simulation and get performance
            performance = self.simulate_drone_response(pid_params, trial.number)
            
            return performance
        # Create and run optimization study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        
        return study.best_params, study.best_value

def main():
    # Path to your MuJoCo XML model
    xml_path = '/home/ws/src/drone_mujoco/model/scene.xml'
    
    # Initialize optimizer
    optimizer = MujocoPIDOptimizer(xml_path)
    
    # Run PID parameter optimization
    best_params, best_performance = optimizer.optimize_pid_parameters()
    
    print("Best PID Parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Performance Score: {best_performance}")

if __name__ == '__main__':
    rclpy.init()
    main()
    rclpy.shutdown()