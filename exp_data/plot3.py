import numpy as np
import matplotlib.pyplot as plt
import math
import os

def wrap_angle(angle):
    """Normalize angle to [-pi, pi] range"""
    return np.arctan2(np.sin(angle), np.cos(angle))

def calculate_velocity(x, y, z, time):
    """Calculate 3D velocity"""
    velocity = np.zeros(len(x))
    for i in range(1, len(x)):
        dt = time[i] - time[i-1]
        if dt > 0:
            dx = x[i] - x[i-1]
            dy = y[i] - y[i-1]
            dz = z[i] - z[i-1]
            velocity[i] = np.sqrt(dx**2 + dy**2 + dz**2) / dt
    return velocity

def calculate_time_per_target(time, targets_hit):
    """Calculate time spent on each target"""
    target_times = []
    current_target = 0
    start_time = time[0]
    
    for i in range(len(targets_hit)):
        if targets_hit[i] > current_target:
            target_time = time[i] - start_time
            target_times.append(target_time)
            current_target = targets_hit[i]
            start_time = time[i]
    
    return target_times

def plot_experiment_3_data(file_path, save_path=None):
    """
    Plot data from Experiment 3 with separate plots for each component.
    """
    # Load data
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    
    # Extract columns
    time = data[:, 0]
    targets_hit = data[:, 1]
    drone_x = data[:, 3]
    drone_y = data[:, 4]
    drone_z = data[:, 5]
    drone_yaw = data[:, 8]
    target_x = data[:, 9]
    target_y = data[:, 10]
    target_z = data[:, 11]
    target_yaw = data[:, 12]
    yaw_error = data[:, 14]
    
    # Wrap angles
    target_yaw_wrapped = np.array([wrap_angle(yaw) for yaw in target_yaw])
    drone_yaw_wrapped = np.array([wrap_angle(yaw) for yaw in drone_yaw])
    
    # Calculate velocity
    velocity = calculate_velocity(drone_x, drone_y, drone_z, time)
    
    # Calculate time per target
    target_times = calculate_time_per_target(time, targets_hit)
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. X Position
    axes[0,0].plot(time, drone_x, 'b-', linewidth=2, label='Drone X')
    axes[0,0].plot(time, target_x, 'r--', linewidth=2, label='Target X', alpha=0.7)
    axes[0,0].set_title("X Position")
    axes[0,0].set_xlabel("Time [s]")
    axes[0,0].set_ylabel("X [m]")
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # 2. Y Position
    axes[0,1].plot(time, drone_y, 'b-', linewidth=2, label='Drone Y')
    axes[0,1].plot(time, target_y, 'r--', linewidth=2, label='Target Y', alpha=0.7)
    axes[0,1].set_title("Y Position")
    axes[0,1].set_xlabel("Time [s]")
    axes[0,1].set_ylabel("Y [m]")
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # 3. Z Position
    axes[0,2].plot(time, drone_z, 'b-', linewidth=2, label='Drone Z')
    axes[0,2].plot(time, target_z, 'r--', linewidth=2, label='Target Z', alpha=0.7)
    axes[0,2].set_title("Z Position")
    axes[0,2].set_xlabel("Time [s]")
    axes[0,2].set_ylabel("Z [m]")
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].legend()
    
    # 4. Velocity with average line
    avg_velocity = np.mean(velocity[velocity > 0])
    axes[1,0].plot(time, velocity, 'purple', linewidth=2, label='3D Velocity')
    axes[1,0].axhline(y=avg_velocity, color='red', linestyle='--', linewidth=2, 
                     label=f'Average: {avg_velocity:.2f} m/s')
    axes[1,0].set_title("Drone Velocity")
    axes[1,0].set_xlabel("Time [s]")
    axes[1,0].set_ylabel("Velocity [m/s]")
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()
    
    # 5. Time per target
    if len(target_times) > 0:
        target_numbers = range(1, len(target_times) + 1)
        axes[1,1].bar(target_numbers, target_times, color='orange', alpha=0.7, label='Time per target')
        axes[1,1].axhline(y=np.mean(target_times), color='red', linestyle='--', 
                         linewidth=2, label=f'Average: {np.mean(target_times):.1f}s')
        axes[1,1].set_title("Time per Target")
        axes[1,1].set_xlabel("Target Number")
        axes[1,1].set_ylabel("Time [s]")
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
    else:
        axes[1,1].text(0.5, 0.5, 'No targets hit', transform=axes[1,1].transAxes, 
                      ha='center', va='center', fontsize=14)
        axes[1,1].set_title("Time per Target")
    
    # 6. YAW TRACKING
    axes[1,2].plot(time, np.degrees(drone_yaw_wrapped), 'b-', linewidth=2, label='Drone Yaw')
    axes[1,2].plot(time, np.degrees(target_yaw_wrapped), 'r-', linewidth=2, label='Target Yaw')
    axes[1,2].set_title("YAW TRACKING")
    axes[1,2].set_xlabel("Time [s]")
    axes[1,2].set_ylabel("Yaw [degrees]")
    axes[1,2].set_ylim(-180, 180)
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].legend()
    

    plt.suptitle(f'Experiment 3 - Analysis\nFile: {os.path.basename(file_path)}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    # Print statistics
    final_targets_hit = int(targets_hit[-1])
    max_velocity = np.max(velocity)
    avg_yaw_error = np.mean(yaw_error)
    max_yaw_error = np.max(yaw_error)
    
    print("\n" + "="*60)
    print("EXPERIMENT 3 - RESULTS")
    print("="*60)
    print(f"Duration: {time[-1]:.1f} seconds")
    print(f"Targets Hit: {final_targets_hit}")
    print(f"Average Velocity: {avg_velocity:.2f} m/s")
    print(f"Max Velocity: {max_velocity:.2f} m/s")
    print(f"Average Yaw Error: {np.degrees(avg_yaw_error):.1f}°")
    print(f"Max Yaw Error: {np.degrees(max_yaw_error):.1f}°")
    if len(target_times) > 0:
        print(f"Average Time per Target: {np.mean(target_times):.1f}s")
        print(f"Fastest Target: {np.min(target_times):.1f}s")
        print(f"Slowest Target: {np.max(target_times):.1f}s")
    print("="*60)


if __name__ == "__main__":
    file_path = "/home/ws/exp_data/rl_exp3_data_1.txt"  
    save_path = "/home/ws/exp_data/exp3_clean_plot_1.png"   
    
    if os.path.exists(file_path):
        plot_experiment_3_data(file_path, save_path=save_path)
    else:
        print(f"File not found: {file_path}")
        print("Make sure to run the experiment first to generate data!")
        
        # List available files
        data_dir = "/home/ws/exp_data"
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.startswith('exp3_data_') and f.endswith('.txt')]
            if files:
                print("\nAvailable experiment files:")
                for f in sorted(files):
                    print(f"  {f}")