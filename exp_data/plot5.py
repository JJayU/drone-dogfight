import numpy as np
import matplotlib.pyplot as plt
import math
import os

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

def calculate_final_energy(m1_power, m2_power, m3_power, m4_power, time):
    """Calculate final energy consumption for each motor and total"""
    dt = np.diff(time)
    dt = np.append(dt, dt[-1])  # Add last dt
    
    # Energy = Power * Time for each motor
    m1_energy = np.cumsum(m1_power * dt)
    m2_energy = np.cumsum(m2_power * dt)
    m3_energy = np.cumsum(m3_power * dt)
    m4_energy = np.cumsum(m4_power * dt)
    
    total_energy = m1_energy + m2_energy + m3_energy + m4_energy
    
    return m1_energy, m2_energy, m3_energy, m4_energy, total_energy

def plot_experiment_5_data(file_path, save_path=None):
    """
    Plot data from Experiment 5 with energy analysis and motor power charts.
    """
    # Load data
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    
    # Extract columns
    time = data[:, 0]
    drone_x = data[:, 1]
    drone_y = data[:, 2]
    drone_z = data[:, 3]
    drone_roll = data[:, 4]
    drone_pitch = data[:, 5]
    drone_yaw = data[:, 6]
    target_x = data[:, 7]
    target_y = data[:, 8]
    target_z = data[:, 9]
    target_direction = data[:, 10]
    distance_to_target = data[:, 11]
    aiming_error = data[:, 12]
    m1_power = data[:, 13]
    m2_power = data[:, 14]
    m3_power = data[:, 15]
    m4_power = data[:, 16]
    cumulative_energy = data[:, 17]
    
    # Calculate velocity
    velocity = calculate_velocity(drone_x, drone_y, drone_z, time)
    
    # Calculate final energy for each motor
    m1_energy, m2_energy, m3_energy, m4_energy, total_energy = calculate_final_energy(
        m1_power, m2_power, m3_power, m4_power, time)
    
    # Calculate total distance traveled
    total_distance = 0
    for i in range(1, len(drone_x)):
        dx = drone_x[i] - drone_x[i-1]
        dy = drone_y[i] - drone_y[i-1]
        dz = drone_z[i] - drone_z[i-1]
        total_distance += np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # 1. XY Trajectory
    axes[0,0].plot(drone_x, drone_y, 'b-', linewidth=2, label='Drone path', alpha=0.8)
    axes[0,0].plot(target_x, target_y, 'r--', linewidth=1, label='Target path', alpha=0.6)
    axes[0,0].scatter(drone_x[0], drone_y[0], color='green', s=100, marker='o', label='Start')
    axes[0,0].scatter(drone_x[-1], drone_y[-1], color='red', s=100, marker='x', label='End')
    axes[0,0].set_title("XY Trajectory - Figure 8 Following")
    axes[0,0].set_xlabel("X [m]")
    axes[0,0].set_ylabel("Y [m]")
    axes[0,0].axis('equal')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend()
    
    # 2. Z Position
    axes[0,1].plot(time, drone_z, 'b-', linewidth=2, label='Drone Z')
    axes[0,1].plot(time, target_z, 'r--', linewidth=2, label='Target Z', alpha=0.7)
    axes[0,1].set_title("Altitude Following")
    axes[0,1].set_xlabel("Time [s]")
    axes[0,1].set_ylabel("Z [m]")
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # 3. Distance to Target
    axes[0,2].plot(time, distance_to_target, 'purple', linewidth=2, label='Distance to target')
    axes[0,2].axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Follow distance (0.5m)')
    axes[0,2].set_title("Target Following Performance")
    axes[0,2].set_xlabel("Time [s]")
    axes[0,2].set_ylabel("Distance [m]")
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].legend()
    
    # 4. Velocity
    avg_velocity = np.mean(velocity[velocity > 0])
    axes[1,0].plot(time, velocity, 'orange', linewidth=2, label='3D Velocity')
    axes[1,0].axhline(y=avg_velocity, color='red', linestyle='--', linewidth=2, 
                     label=f'Average: {avg_velocity:.2f} m/s')
    axes[1,0].set_title("Drone Velocity")
    axes[1,0].set_xlabel("Time [s]")
    axes[1,0].set_ylabel("Velocity [m/s]")
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()
    
    # 5. Motor Powers
    axes[1,1].plot(time, m1_power, 'red', linewidth=1.5, label='Motor 1', alpha=0.8)
    axes[1,1].plot(time, m2_power, 'blue', linewidth=1.5, label='Motor 2', alpha=0.8)
    axes[1,1].plot(time, m3_power, 'green', linewidth=1.5, label='Motor 3', alpha=0.8)
    axes[1,1].plot(time, m4_power, 'orange', linewidth=1.5, label='Motor 4', alpha=0.8)
    axes[1,1].set_title("Individual Motor Powers")
    axes[1,1].set_xlabel("Time [s]")
    axes[1,1].set_ylabel("Motor Power")
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend()
    
    # 6. Total Power
    total_power = m1_power + m2_power + m3_power + m4_power
    avg_total_power = np.mean(total_power)
    axes[1,2].plot(time, total_power, 'purple', linewidth=2, label='Total Power')
    axes[1,2].axhline(y=avg_total_power, color='red', linestyle='--', linewidth=2, 
                     label=f'Average: {avg_total_power:.2f}')
    axes[1,2].set_title("Total Motor Power")
    axes[1,2].set_xlabel("Time [s]")
    axes[1,2].set_ylabel("Total Power")
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].legend()
    
    # 7. Energy Consumption (Cumulative)
    axes[2,0].plot(time, m1_energy, 'red', linewidth=2, label='Motor 1', alpha=0.8)
    axes[2,0].plot(time, m2_energy, 'blue', linewidth=2, label='Motor 2', alpha=0.8)
    axes[2,0].plot(time, m3_energy, 'green', linewidth=2, label='Motor 3', alpha=0.8)
    axes[2,0].plot(time, m4_energy, 'orange', linewidth=2, label='Motor 4', alpha=0.8)
    axes[2,0].plot(time, total_energy, 'black', linewidth=3, label='Total Energy')
    axes[2,0].set_title("Cumulative Energy Consumption")
    axes[2,0].set_xlabel("Time [s]")
    axes[2,0].set_ylabel("Energy")
    axes[2,0].grid(True, alpha=0.3)
    axes[2,0].legend()
    
    # 8. Final Energy Bar Chart
    final_energies = [m1_energy[-1], m2_energy[-1], m3_energy[-1], m4_energy[-1]]
    motor_names = ['Motor 1', 'Motor 2', 'Motor 3', 'Motor 4']
    colors = ['red', 'blue', 'green', 'orange']
    
    bars = axes[2,1].bar(motor_names, final_energies, color=colors, alpha=0.7)
    axes[2,1].set_title("Final Energy Consumption per Motor")
    axes[2,1].set_ylabel("Total Energy")
    axes[2,1].grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, energy in zip(bars, final_energies):
        height = bar.get_height()
        axes[2,1].text(bar.get_x() + bar.get_width()/2., height + 0.01*max(final_energies),
                      f'{energy:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 9. Energy Efficiency
    efficiency = total_distance / total_energy[-1] if total_energy[-1] > 0 else 0
    power_per_meter = total_energy[-1] / total_distance if total_distance > 0 else 0
    
    axes[2,2].text(0.1, 0.8, f"ENERGY ANALYSIS", fontsize=16, fontweight='bold', 
                   transform=axes[2,2].transAxes)
    axes[2,2].text(0.1, 0.7, f"Total Distance: {total_distance:.2f} m", fontsize=12, 
                   transform=axes[2,2].transAxes)
    axes[2,2].text(0.1, 0.6, f"Total Energy: {total_energy[-1]:.2f}", fontsize=12, 
                   transform=axes[2,2].transAxes)
    axes[2,2].text(0.1, 0.5, f"Energy Efficiency: {efficiency:.3f} m/unit", fontsize=12, 
                   transform=axes[2,2].transAxes)
    axes[2,2].text(0.1, 0.4, f"Power per Meter: {power_per_meter:.3f} unit/m", fontsize=12, 
                   transform=axes[2,2].transAxes)
    axes[2,2].text(0.1, 0.3, f"Average Power: {avg_total_power:.2f} unit/s", fontsize=12, 
                   transform=axes[2,2].transAxes)
    axes[2,2].text(0.1, 0.2, f"Flight Time: {time[-1]:.1f} seconds", fontsize=12, 
                   transform=axes[2,2].transAxes)
    
    # Add motor energy distribution pie chart
    axes[2,2].pie(final_energies, labels=motor_names, colors=colors, autopct='%1.1f%%', 
                 startangle=90, center=(0.7, 0.5), radius=0.3)
    axes[2,2].set_title("Motor Energy Distribution", fontsize=10)
    axes[2,2].axis('off')
    
    # Overall title
    plt.suptitle(f'Experiment 5 - Figure-8 Following with Energy Analysis\nFile: {os.path.basename(file_path)}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    
    # Print detailed statistics
    print("\n" + "="*70)
    print("EXPERIMENT 5 - FIGURE-8 FOLLOWING ANALYSIS")
    print("="*70)
    print(f"Flight Duration: {time[-1]:.1f} seconds")
    print(f"Total Distance Traveled: {total_distance:.2f} meters")
    print(f"Average Velocity: {avg_velocity:.2f} m/s")
    print(f"Max Velocity: {np.max(velocity):.2f} m/s")
    print(f"Average Distance to Target: {np.mean(distance_to_target):.3f} m")
    print(f"Max Distance to Target: {np.max(distance_to_target):.3f} m")
    print(f"Average Aiming Error: {np.degrees(np.mean(aiming_error)):.1f}°")
    print("\nENERGY CONSUMPTION:")
    print(f"Motor 1 Final Energy: {m1_energy[-1]:.3f}")
    print(f"Motor 2 Final Energy: {m2_energy[-1]:.3f}")
    print(f"Motor 3 Final Energy: {m3_energy[-1]:.3f}")
    print(f"Motor 4 Final Energy: {m4_energy[-1]:.3f}")
    print(f"Total Energy Consumed: {total_energy[-1]:.3f}")
    print(f"Average Total Power: {avg_total_power:.3f} units/s")
    print(f"Energy Efficiency: {efficiency:.3f} meters/energy_unit")
    print(f"Power per Meter: {power_per_meter:.3f} energy_units/meter")
    print("="*70)


if __name__ == "__main__":
    file_path = "/home/ws/exp_data/exp5_data_1.txt"  
    save_path = "/home/ws/exp_data/exp5_energy_plot_1.png"   
    
    if os.path.exists(file_path):
        plot_experiment_5_data(file_path, save_path=save_path)
    else:
        print(f"File not found: {file_path}")
        print("Make sure to run the experiment first to generate data!")
        
        # List available files
        data_dir = "/home/ws/exp_data"
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.startswith('exp5_data_') and f.endswith('.txt')]
            if files:
                print("\nAvailable experiment files:")
                for f in sorted(files):
                    print(f"  {f}")
            else:
                print("No exp5 data files found.")