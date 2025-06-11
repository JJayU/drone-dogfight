import numpy as np
import matplotlib.pyplot as plt
import math
import os

def plot_experiment_3_data(file_path, save_path=None):
    """
    Plot data from Experiment 3 with comprehensive analysis.
    Expected file format:
    time, targets_hit, total_targets, drone_x, drone_y, drone_z,
    drone_roll, drone_pitch, drone_yaw, target_x, target_y, target_z,
    distance_to_target, is_hitting
    """
    # Load data
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    
    # Extract columns
    time = data[:, 0]
    targets_hit = data[:, 1]
    total_targets = data[:, 2]
    drone_x = data[:, 3]
    drone_y = data[:, 4]
    drone_z = data[:, 5]
    drone_roll = data[:, 6]
    drone_pitch = data[:, 7]
    drone_yaw = data[:, 8]
    target_x = data[:, 9]
    target_y = data[:, 10]
    target_z = data[:, 11]
    distance_to_target = data[:, 12]
    is_hitting = data[:, 13]
    
    # Calculate metrics
    pos_error = np.sqrt((drone_x - target_x)**2 + (drone_y - target_y)**2 + (drone_z - target_z)**2)
    velocity = np.zeros_like(time)
    acceleration = np.zeros_like(time)
    
    # Calculate velocity and acceleration
    for i in range(1, len(time)):
        dt = time[i] - time[i-1]
        if dt > 0:
            dx = drone_x[i] - drone_x[i-1]
            dy = drone_y[i] - drone_y[i-1]
            dz = drone_z[i] - drone_z[i-1]
            velocity[i] = np.sqrt(dx**2 + dy**2 + dz**2) / dt
            
            if i > 1:
                acceleration[i] = (velocity[i] - velocity[i-1]) / dt
    
    # Calculate target hit timing
    hit_times = []
    hit_indices = np.where(np.diff(targets_hit.astype(int)) > 0)[0] + 1
    
    if len(hit_indices) > 0:
        # Calculate time to hit each target
        target_start_time = 0
        for hit_idx in hit_indices:
            hit_time = time[hit_idx]
            time_to_hit = hit_time - target_start_time
            hit_times.append(time_to_hit)
            target_start_time = hit_time
    
    # Calculate additional metrics
    final_targets_hit = int(targets_hit[-1])
    final_total_targets = int(total_targets[-1])
    avg_time_to_hit = np.mean(hit_times) if hit_times else 0
    success_rate = (final_targets_hit / final_total_targets * 100) if final_total_targets > 0 else 0
    
    # Movement efficiency metrics
    total_distance = np.sum(np.sqrt(np.diff(drone_x)**2 + np.diff(drone_y)**2 + np.diff(drone_z)**2))
    avg_velocity = np.mean(velocity[velocity > 0])
    max_velocity = np.max(velocity)
    avg_error = np.mean(pos_error)
    min_error = np.min(pos_error)
    max_error = np.max(pos_error)
    
    # Time spent hitting vs moving
    hitting_time = np.sum(is_hitting) * (time[1] - time[0]) if len(time) > 1 else 0
    hitting_percentage = (hitting_time / time[-1]) * 100
    
    # Plotting - Enhanced layout
    fig = plt.figure(figsize=(18, 14))
    
    # Create grid layout
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.8], hspace=0.3, wspace=0.3)
    
    # Top-down trajectory view
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(drone_x, drone_y, c=time, cmap='viridis', s=10, alpha=0.7)
    ax1.scatter(target_x[::50], target_y[::50], c='red', s=50, marker='x', label='Targets')
    ax1.set_title("XY Trajectory (colored by time)")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='Time [s]')
    
    # 3D trajectory view
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax2.plot(drone_x, drone_y, drone_z, 'b-', alpha=0.7, linewidth=1)
    ax2.scatter(target_x[::50], target_y[::50], target_z[::50], c='red', s=30)
    ax2.set_title("3D Trajectory")
    ax2.set_xlabel("X [m]")
    ax2.set_ylabel("Y [m]")
    ax2.set_zlabel("Z [m]")
    
    # Targets hit over time
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(time, targets_hit, 'g-', linewidth=2, label='Targets Hit')
    ax3.plot(time, total_targets, 'r--', label='Total Targets')
    ax3.set_title("Target Progress")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Count")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Position tracking
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(time, drone_x, 'b-', label='Drone X', alpha=0.8)
    ax4.plot(time, target_x, 'r--', label='Target X', alpha=0.6)
    ax4.set_title("X Position Tracking")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("X [m]")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(time, drone_y, 'b-', label='Drone Y', alpha=0.8)
    ax5.plot(time, target_y, 'r--', label='Target Y', alpha=0.6)
    ax5.set_title("Y Position Tracking")
    ax5.set_xlabel("Time [s]")
    ax5.set_ylabel("Y [m]")
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(time, drone_z, 'b-', label='Drone Z', alpha=0.8)
    ax6.plot(time, target_z, 'r--', label='Target Z', alpha=0.6)
    ax6.set_title("Z Position Tracking")
    ax6.set_xlabel("Time [s]")
    ax6.set_ylabel("Z [m]")
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # Error and performance metrics
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.fill_between(time, 0, pos_error, alpha=0.5, color='orange', label='Position Error')
    ax7.axhline(y=0.3, color='red', linestyle='--', label='Hit Threshold')
    ax7.fill_between(time, 0, is_hitting * np.max(pos_error), alpha=0.3, color='green', label='Hitting Target')
    ax7.set_title("Position Error & Target Hitting")
    ax7.set_xlabel("Time [s]")
    ax7.set_ylabel("Error [m]")
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # Velocity profile
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(time, velocity, 'purple', alpha=0.8, label='Velocity')
    ax8.axhline(y=avg_velocity, color='red', linestyle='--', alpha=0.7, label=f'Avg: {avg_velocity:.2f} m/s')
    ax8.set_title("Velocity Profile")
    ax8.set_xlabel("Time [s]")
    ax8.set_ylabel("Velocity [m/s]")
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    
    # Target hit timing
    ax9 = fig.add_subplot(gs[2, 2])
    if hit_times:
        ax9.bar(range(1, len(hit_times) + 1), hit_times, alpha=0.7, color='green')
        ax9.axhline(y=avg_time_to_hit, color='red', linestyle='--', 
                   label=f'Avg: {avg_time_to_hit:.2f}s')
        ax9.set_title("Time to Hit Each Target")
        ax9.set_xlabel("Target Number")
        ax9.set_ylabel("Time [s]")
        ax9.grid(True, alpha=0.3)
        ax9.legend()
    else:
        ax9.text(0.5, 0.5, 'No targets hit', ha='center', va='center', transform=ax9.transAxes)
        ax9.set_title("Time to Hit Each Target")
    
    # Summary statistics panel
    ax10 = fig.add_subplot(gs[3, :])
    ax10.axis('off')
    

    plt.suptitle(f'Experiment 3 - Target Hitting Analysis\nFile: {os.path.basename(file_path)}', 
                 fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Enhanced plot saved to {save_path}")
    
    plt.show()
    
    # Print detailed analysis
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"📊 Overall Performance:")
    print(f"   • Duration: {time[-1]:.2f} seconds")
    print(f"   • Targets Hit: {final_targets_hit}")
    
    print(f"\n⏱️  Target Hitting Timing:")
    if hit_times:
        print(f"   • Average Time to Hit: {avg_time_to_hit:.2f} seconds")
        print(f"   • Fastest Hit: {min(hit_times):.2f} seconds")
        print(f"   • Slowest Hit: {max(hit_times):.2f} seconds")
        print(f"   • Hit Time Std Dev: {np.std(hit_times):.2f} seconds")
    else:
        print("   • No targets were successfully hit")
    
    print(f"\n🚁 Movement Analysis:")
    print(f"   • Total Distance Traveled: {total_distance:.2f} meters")
    print(f"   • Average Velocity: {avg_velocity:.2f} m/s")
    print(f"   • Maximum Velocity: {max_velocity:.2f} m/s")
    print(f"   • Time Spent Hitting Targets: {hitting_percentage:.1f}% ({hitting_time:.1f}s)")
    
    print(f"\n🎯 Accuracy Metrics:")
    print(f"   • Average Position Error: {avg_error:.3f} meters")
    print(f"   • Best Accuracy: {min_error:.3f} meters")
    print(f"   • Worst Accuracy: {max_error:.3f} meters")
    print(f"   • Error Standard Deviation: {np.std(pos_error):.3f} meters")
    

if __name__ == "__main__":
    file_path = "/home/ws/exp_data/rl_exp3_data_1.txt"  
    save_path = "/home/ws/exp_data/exp3_enhanced_plot_1.png"   
    
    if os.path.exists(file_path):
        plot_experiment_3_data(file_path, save_path=save_path)
    else:
        print(f"File not found: {file_path}")
        print("Make sure to run the experiment first to generate data!")