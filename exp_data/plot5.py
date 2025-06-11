import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_experiment_5_data(file_path, save_path=None):
    """
    Plot data from Experiment 5 - Drone following target in figure-8 and aiming at rear.
    
    Updated data format:
    time, drone_x, drone_y, drone_z, drone_roll, drone_pitch, drone_yaw,
    target_x, target_y, target_z, target_direction, distance_to_target, 
    aiming_error, m1_power, m2_power, m3_power, m4_power, cumulative_energy
    """
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
    target_dir = data[:, 10]
    
    distance = data[:, 11]
    aiming_error = data[:, 12]
    
    # Motor powers and energy (if available)
    has_motor_data = data.shape[1] >= 18
    if has_motor_data:
        m1_power = data[:, 13]
        m2_power = data[:, 14]
        m3_power = data[:, 15]
        m4_power = data[:, 16]
        cumulative_energy = data[:, 17]
        
        # Calculate instantaneous power
        total_power = np.abs(m1_power) + np.abs(m2_power) + np.abs(m3_power) + np.abs(m4_power)
        power_squared = m1_power**2 + m2_power**2 + m3_power**2 + m4_power**2
    
    # === Create figure with subplots ===
    fig = plt.figure(figsize=(20, 14))
    
    # === Plot 3D Trajectories ===
    ax1 = fig.add_subplot(331, projection='3d')
    # ax1.plot(drone_x, drone_y, drone_z, label="Drone", color='blue', linewidth=2)
    ax1.plot(target_x, target_y, target_z, label="Target", color='red', linewidth=2)
    ax1.scatter(drone_x[0], drone_y[0], drone_z[0], color='blue', s=100, marker='o', label='Drone Start')
    ax1.scatter(target_x[0], target_y[0], target_z[0], color='red', s=100, marker='s', label='Target Start')
    ax1.set_title("3D Trajectory")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.set_zlabel("Z [m]")
    ax1.legend()
    ax1.grid(True)
    
    # === Distance to Target Over Time ===
    plt.subplot(332)
    plt.plot(time, distance, color='green', linewidth=2)
    plt.axhline(y=np.mean(distance), color='green', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(distance):.2f}m')
    plt.title("Distance to Target Over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.legend()
    plt.grid(True)
    
    # === Aiming Error Over Time ===
    plt.subplot(333)
    plt.plot(time, np.degrees(aiming_error), color='purple', linewidth=2)
    plt.axhline(y=np.degrees(np.mean(aiming_error)), color='purple', linestyle='--', alpha=0.7, 
                label=f'Mean: {np.degrees(np.mean(aiming_error)):.2f}°')
    # Add threshold line for "good aiming"
    plt.axhline(y=5.7, color='red', linestyle=':', alpha=0.7, label='Threshold: 5.7°')
    plt.title("Aiming Error (Yaw) Over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Aiming Error [deg]")
    plt.legend()
    plt.grid(True)
    
    # === Z Positions Over Time ===
    plt.subplot(334)
    plt.plot(time, drone_z, label="Drone Z", color='blue', linewidth=2)
    plt.plot(time, target_z, label="Target Z", color='red', linestyle='--', linewidth=2)
    plt.title("Z Position Tracking")
    plt.xlabel("Time [s]")
    plt.ylabel("Z Position [m]")
    plt.legend()
    plt.grid(True)
    
    # === XY Trajectory (Top View) ===
    plt.subplot(335)
    plt.plot(drone_x, drone_y, label="Drone", color='blue', linewidth=2)
    plt.plot(target_x, target_y, label="Target", color='red', linewidth=2)
    plt.scatter(drone_x[0], drone_y[0], color='blue', s=100, marker='o', label='Start')
    plt.scatter(target_x[0], target_y[0], color='red', s=100, marker='s')
    plt.title("XY Trajectory (Top View)")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    if has_motor_data:
        # === Motor Powers Over Time ===
        plt.subplot(336)
        plt.plot(time, m1_power, label="M1", alpha=0.8)
        plt.plot(time, m2_power, label="M2", alpha=0.8)
        plt.plot(time, m3_power, label="M3", alpha=0.8)
        plt.plot(time, m4_power, label="M4", alpha=0.8)
        plt.title("Motor Powers Over Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Motor Power")
        plt.legend()
        plt.grid(True)
        
        # === Total Power and Energy ===
        plt.subplot(337)
        plt.plot(time, total_power, label="Total Power", color='orange', linewidth=2)
        plt.title("Total Motor Power Over Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Total Power")
        plt.legend()
        plt.grid(True)
        
        # === Cumulative Energy ===
        plt.subplot(338)
        plt.plot(time, cumulative_energy, label="Cumulative Energy", color='red', linewidth=2)
        plt.title("Cumulative Energy Consumption")
        plt.xlabel("Time [s]")
        plt.ylabel("Energy")
        plt.legend()
        plt.grid(True)
        
        # === Power Squared (Energy Rate) ===
        plt.subplot(339)
        plt.plot(time, power_squared, label="Power² (Energy Rate)", color='darkred', linewidth=2)
        plt.title("Instantaneous Energy Rate (u²)")
        plt.xlabel("Time [s]")
        plt.ylabel("Power² [u²]")
        plt.legend()
        plt.grid(True)
    else:
        # === Attitude Over Time (if no motor data) ===
        plt.subplot(336)
        plt.plot(time, np.degrees(drone_roll), label="Roll", alpha=0.8)
        plt.plot(time, np.degrees(drone_pitch), label="Pitch", alpha=0.8)
        plt.plot(time, np.degrees(drone_yaw), label="Yaw", alpha=0.8)
        plt.title("Drone Attitude Over Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Angle [deg]")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Experiment 5: Figure-8 Target Tracking & Aiming Analysis", fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # === COMPLETE STATISTICS ===
    print("\n" + "="*60)
    print("           EXPERIMENT 5 - COMPLETE ANALYSIS")
    print("="*60)
    
    # Basic stats
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   Total experiment time: {time[-1]:.2f} seconds")
    print(f"   Data points collected: {len(time)}")
    print(f"   Sampling rate: {1/np.mean(np.diff(time)):.1f} Hz")
    
    # Distance tracking
    print(f"\n🎯 DISTANCE TRACKING:")
    print(f"   Average distance to target: {np.mean(distance):.3f} m")
    print(f"   Min distance: {np.min(distance):.3f} m")
    print(f"   Max distance: {np.max(distance):.3f} m")
    print(f"   Distance std deviation: {np.std(distance):.3f} m")
    
    # Aiming performance
    print(f"\n🎯 AIMING PERFORMANCE:")
    avg_aiming_error_deg = np.degrees(np.mean(aiming_error))
    max_aiming_error_deg = np.degrees(np.max(aiming_error))
    rmse_aiming_deg = np.degrees(np.sqrt(np.mean(aiming_error**2)))
    
    print(f"   Average aiming error: {avg_aiming_error_deg:.2f}°")
    print(f"   Max aiming error: {max_aiming_error_deg:.2f}°")
    print(f"   RMSE aiming error: {rmse_aiming_deg:.2f}°")
    print(f"   RMSE aiming error: {np.sqrt(np.mean(aiming_error**2)):.4f} rad")
    
    # Laser hit analysis
    dt = np.mean(np.diff(time))
    aiming_threshold_rad = np.radians(1) 
    accurate_shots = aiming_error < aiming_threshold_rad
    hit_count = np.sum(accurate_shots)
    hit_time_total = hit_count * dt
    hit_percentage = (hit_count / len(aiming_error)) * 100
    
    print(f"\n🔫 LASER TARGETING (threshold: {np.degrees(aiming_threshold_rad):.1f}°):")
    print(f"   Total hit time: {hit_time_total:.2f} s")
    print(f"   Hit percentage: {hit_percentage:.1f}%")
    print(f"   Number of accurate shots: {hit_count}/{len(aiming_error)}")
    
    hit_sequences = []
    current_seq = 0
    for hit in accurate_shots:
        if hit:
            current_seq += 1
        else:
            if current_seq > 0:
                hit_sequences.append(current_seq)
            current_seq = 0
    if current_seq > 0:
        hit_sequences.append(current_seq)
    
    longest_hit_sequence = max(hit_sequences) * dt if hit_sequences else 0
    print(f"   Longest continuous hit: {longest_hit_sequence:.2f} s")
    
    if has_motor_data:
        final_energy = cumulative_energy[-1]
        avg_power = np.mean(total_power)
        max_power = np.max(total_power)
        avg_power_squared = np.mean(power_squared)
        
        print(f"\n⚡ ENERGY CONSUMPTION:")
        print(f"   Total energy (sum u²): {final_energy:.6f}")
        print(f"   Average total power: {avg_power:.3f}")
        print(f"   Max total power: {max_power:.3f}")
        print(f"   Average power² (u²): {avg_power_squared:.6f}")
        print(f"   Energy efficiency: {final_energy/time[-1]:.6f} energy/second")
    
    print(f"\n" + "="*60)
    print("              📋 KEY METRICS SUMMARY")
    print("="*60)
    print(f"Błąd średniokwadratowy celowania: {np.sqrt(np.mean(aiming_error**2)):.4f} rad")
    print(f"Błąd średniokwadratowy celowania: {rmse_aiming_deg:.2f}°")
    
    if has_motor_data:
        print(f"Zużycie energii (suma u²): {final_energy:.6f}")
    else:
        print("Zużycie energii: BRAK DANYCH - dodaj tracking mocy silników")
    
    print(f"Czas trafienia laserem w cel: {hit_time_total:.2f} s ({hit_percentage:.1f}%)")
    print("="*60)
    
    return {
        'rmse_aiming_rad': np.sqrt(np.mean(aiming_error**2)),
        'rmse_aiming_deg': rmse_aiming_deg,
        'total_energy': final_energy if has_motor_data else None,
        'hit_time_total': hit_time_total,
        'hit_percentage': hit_percentage,
        'longest_hit_sequence': longest_hit_sequence,
        'avg_distance': np.mean(distance),
        'max_distance': np.max(distance),
        'min_distance': np.min(distance)
    }

if __name__ == "__main__":
    data_file = "/home/ws/exp_data/rl_exp5_data_1.txt"
    save_file = "/home/ws/exp_data/rl_exp5_plot_1.png"
    
    try:
        results = plot_experiment_5_data(data_file, save_file)
        print(f"\nAnalysis complete! Results saved to: {save_file}")
    except FileNotFoundError:
        print(f"Data file not found: {data_file}")
        print("Make sure to run the experiment first to generate data.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()