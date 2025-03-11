#!/usr/bin/env python3
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
import os

def plot_flight_data(data_file='flight_data.pkl'):
    """Generate plots from saved flight data"""
    
    if not os.path.exists(data_file):
        print(f"Error: Could not find {data_file}")
        print("Make sure to run the drone control node first to generate flight data.")
        return
        
    print(f"Loading flight data from {data_file}...")
    try:
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Extract data
    position = data['position']
    attitude = data['attitude']
    target = data['target']
    x_pid = data['x_pid']
    y_pid = data['y_pid']
    z_pid = data['z_pid']
    roll_pid = data['roll_pid']
    pitch_pid = data['pitch_pid']
    yaw_pid = data['yaw_pid']
    
    # Create main figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 3D Trajectory Plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(position['x'], position['y'], position['z'], 'b-', label='Actual Path')
    ax1.plot(target['x'], target['y'], target['z'], 'r--', label='Target Path')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_zlabel('Z Position (m)')
    ax1.set_title('Drone 3D Trajectory')
    ax1.legend()
    
    # 2. Position Errors Plot
    ax2 = fig.add_subplot(222)
    ax2.plot(x_pid['time'], x_pid['error'], 'r-', label='X Error')
    ax2.plot(y_pid['time'], y_pid['error'], 'g-', label='Y Error')
    ax2.plot(z_pid['time'], z_pid['error'], 'b-', label='Z Error')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error (m)')
    ax2.set_title('Position Control Errors')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Attitude Errors Plot
    ax3 = fig.add_subplot(223)
    ax3.plot(roll_pid['time'], roll_pid['error'], 'r-', label='Roll Error')
    ax3.plot(pitch_pid['time'], pitch_pid['error'], 'g-', label='Pitch Error')
    ax3.plot(yaw_pid['time'], yaw_pid['error'], 'b-', label='Yaw Error')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Error (rad)')
    ax3.set_title('Attitude Control Errors')
    ax3.grid(True)
    ax3.legend()
    
    # 4. Position Over Time Plot
    ax4 = fig.add_subplot(224)
    ax4.plot(position['time'], position['x'], 'r-', label='X Actual')
    ax4.plot(position['time'], position['y'], 'g-', label='Y Actual')
    ax4.plot(position['time'], position['z'], 'b-', label='Z Actual')
    ax4.plot(target['time'], target['x'], 'r--', label='X Target')
    ax4.plot(target['time'], target['y'], 'g--', label='Y Target')
    ax4.plot(target['time'], target['z'], 'b--', label='Z Target')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position (m)')
    ax4.set_title('Position vs. Time')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('drone_flight_analysis.png')
    print("Saved overview plot as 'drone_flight_analysis.png'")
    
    # Create detailed PID controller plots
    for pid_data, name in zip(
        [x_pid, y_pid, z_pid, roll_pid, pitch_pid, yaw_pid],
        ['X Position', 'Y Position', 'Height', 'Roll', 'Pitch', 'Yaw']
    ):
        _plot_pid_details(name, pid_data)
    
    print("All plots have been generated. Displaying plots...")
    plt.show()

def _plot_pid_details(title, pid_data):
    """Create detailed plots for a single PID controller"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Setpoint vs Actual (implied from error)
    actual_values = []
    for i in range(len(pid_data['setpoint'])):
        actual_values.append(pid_data['setpoint'][i] - pid_data['error'][i])
        
    ax1.plot(pid_data['time'], pid_data['setpoint'], 'r--', label='Setpoint')
    ax1.plot(pid_data['time'], actual_values, 'b-', label='Actual')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Value')
    ax1.set_title(f'{title} - Setpoint vs Actual')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: Error and Control Output
    ax2.plot(pid_data['time'], pid_data['error'], 'r-', label='Error')
    ax2.plot(pid_data['time'], pid_data['output'], 'g-', label='Control Output')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Value')
    ax2.set_title(f'{title} - Error and Control Output')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    filename = f"{title.replace(' ', '_').lower()}_details.png"
    plt.savefig(filename)
    print(f"Saved {title} plot as '{filename}'")

if __name__ == '__main__':
    data_file = 'flight_data.pkl'
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    
    plot_flight_data(data_file)