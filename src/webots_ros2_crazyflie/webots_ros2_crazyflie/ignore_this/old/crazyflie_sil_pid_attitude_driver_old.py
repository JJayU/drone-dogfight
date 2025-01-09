# Copyright 2023 Matthew Lock.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bitcraze Crazyflie driver."""

import rclpy
import math

from geometry_msgs.msg import PoseStamped

import cffirmware as firm

# ============================= SPLINE STUFF ============================= 
import matplotlib.pyplot as plt
import numpy as np
import splines
points1 = [
    (0      , 0 ,       1.0     ),
    (0      , 2.3,      0.8     ),
    (1      , 1,        1       ),
    (4      , 1.3,      1.5     ),
    (3.8    , -0.2,     0.5     ),
    (2.5    , 0.1,      0.0     ),
]
distances = np.linalg.norm(np.diff(points1, axis=0), axis=1)
times = np.concatenate([[0], np.cumsum(np.sqrt(distances))])
s1 = splines.CatmullRom(points1, alpha=0.5)
times1 = np.linspace(0, times[-1], 1000)
# x = s1.evaluate(times1)[:,0]
# y = s1.evaluate(times1)[:,1]
# z = s1.evaluate(times1)[:,2]
time_Scale = 10
# ============================= SPLINE STUFF ============================= 

class CrazyflieDriver:
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot
        self.__timestep = int(self.__robot.getBasicTimeStep())

        # Sensors
        self.__gps = self.__robot.getDevice('gps')
        self.__gyro = self.__robot.getDevice('gyro')
        self.__imu = self.__robot.getDevice('inertial_unit')

        # Propellers
        self.m1_motor = self.__robot.getDevice('m1_motor')
        self.m2_motor = self.__robot.getDevice('m2_motor')
        self.m3_motor = self.__robot.getDevice('m3_motor')
        self.m4_motor = self.__robot.getDevice('m4_motor')
        
        self.__propellers = [
            self.m1_motor,
            self.m2_motor,
            self.m3_motor,
            self.m4_motor
        ]
        
        for propeller in self.__propellers:
            propeller.setPosition(float('inf'))
            propeller.setVelocity(0)
            
        # Firmware initialization
        firm.controllerPidInit()
        self.state      = firm.state_t()
        self.sensors    = firm.sensorData_t()
        self.setpoint   = firm.setpoint_t()
        self.control    = firm.control_t()
        self.tick       = 100 #this value makes sure that the position controller and attitude controller are always always initiated
        
        # ROS interface
        rclpy.init(args=None)
        self.__node = rclpy.create_node('crazyflie_sil_pid_attitude_driver')
        
        # Pose publisher
        self.pose_publisher = self.__node.create_publisher(
            PoseStamped,
            'pose',
            1
        )
        self.goal_pose_publisher = self.__node.create_publisher(
            PoseStamped,
            'goal_pose',
            1
        )
        
        # States
        self.go_to_start_bool   = True
        self.hovering_time      = 0
        self.start_mission_time = 0
        self.last_time          = self.__robot.getTime()
        self.__node.get_logger().info(f"{str(self.__timestep)}")
        
        #roslog info
        self.logger = self.__node.get_logger()
        
    def go_to_pose(self, des_x=0, des_y=0, des_z=1, des_yaw=0):
        
        yawDesired = 0
        
        ## Fill in Setpoints
        self.setpoint.mode.z            =   firm.modeAbs
        self.setpoint.mode.y            =   firm.modeAbs
        self.setpoint.mode.x            =   firm.modeAbs
        # self.setpoint.mode.yaw          =   firm.modeDisable
        # self.setpoint.mode.roll         =   firm.modeDisable
        # self.setpoint.mode.pitch        =   firm.modeDisable
        
        self.setpoint.position.x        = des_x
        self.setpoint.position.y        = des_y
        self.setpoint.position.z        = des_z
        
        # self.setpoint.attitudeRate.roll     = 0
        # self.setpoint.attitudeRate.pitch    = 0
        # self.setpoint.attitudeRate.yaw      = 0
        
        self.setpoint.velocity_body     = False
        
        ## Firmware PID bindings
        firm.controllerPid(
            self.control, 
            self.setpoint,
            self.sensors,
            self.state,
            self.tick
        )
        
        ## Motor commands
        cmd_roll    =   math.radians(self.control.roll)
        cmd_pitch   =   math.radians(self.control.pitch)
        cmd_yaw     = - math.radians(self.control.yaw)
        cmd_thrust  =   self.control.thrust
        
        ## Motor mixing
        motorPower_m1 =  cmd_thrust - cmd_roll + cmd_pitch + cmd_yaw
        motorPower_m2 =  cmd_thrust - cmd_roll - cmd_pitch - cmd_yaw
        motorPower_m3 =  cmd_thrust + cmd_roll - cmd_pitch + cmd_yaw
        motorPower_m4 =  cmd_thrust + cmd_roll + cmd_pitch - cmd_yaw
        
        scaling = 1000 ##Todo, remove necessity of this scaling (SI units in firmware)
        self.m1_motor.setVelocity(- motorPower_m1/scaling)
        self.m2_motor.setVelocity(  motorPower_m2/scaling)
        self.m3_motor.setVelocity(- motorPower_m3/scaling)
        self.m4_motor.setVelocity(  motorPower_m4/scaling)

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)
        
        ## Get measurements
        x, y, z                                         = self.__gps.getValues()
        vx, vy, vz                                      = self.__gps.getSpeedVector()
        roll, pitch, yaw                                = self.__imu.getRollPitchYaw()
        roll_velocity, pitch_velocity, yaw_velocity     = self.__gyro.getValues()
                
        if math.isnan(vx):
            return
        
        ## Put measurement in state estimate
        self.state.attitude.roll    =   math.degrees(roll)
        self.state.attitude.pitch   = - math.degrees(pitch)
        self.state.attitude.yaw     =   math.degrees(yaw)
        self.state.position.x       =   x
        self.state.position.y       =   y
        self.state.position.z       =   z
        self.state.velocity.x       =   vx
        self.state.velocity.y       =   vy
        self.state.velocity.z       =   vz
            
        # Put gyro in sensor data
        self.sensors.gyro.x = math.degrees(roll_velocity)
        self.sensors.gyro.y = math.degrees(pitch_velocity)
        self.sensors.gyro.z = math.degrees(yaw_velocity)
        
        # publish pose
        pose = PoseStamped()
        pose.header.stamp = self.__node.get_clock().now().to_msg()
        pose.header.frame_id = 'map'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = roll
        pose.pose.orientation.y = pitch
        pose.pose.orientation.z = yaw
        pose.pose.orientation.w = 1.0
        self.pose_publisher.publish(pose)
        
        if self.hovering_time<7:
            self.go_to_pose()
            
            dist_to_start = math.sqrt((x - 0)**2 + (y - 0)**2 + (z - 1)**2)
            
            if dist_to_start < 0.15:
                time_passed = self.__robot.getTime() - self.last_time
                self.hovering_time += time_passed     
                self.last_time = self.__robot.getTime()
            
            else :
                self.hovering_time = 0
                self.last_time = self.__robot.getTime()
            
            self.logger.info(f"Hovering time: {self.hovering_time}")
        
        else:
            if self.go_to_start_bool:
                self.start_mission_time = self.__robot.getTime()
                self.go_to_start_bool = False
            
            time_passed = self.__robot.getTime() - self.start_mission_time
            
            if time_passed < time_Scale*times1[-1]:
                x = s1.evaluate(time_passed/time_Scale)[0]
                y = s1.evaluate(time_passed/time_Scale)[1]
                z = s1.evaluate(time_passed/time_Scale)[2]
                self.go_to_pose(x, y, z)
                
                pose = PoseStamped()
                pose.header.stamp = self.__node.get_clock().now().to_msg()
                pose.header.frame_id = 'map'
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = z
                self.goal_pose_publisher.publish(pose)
                
            else:
                self.logger.info(f"Mission completed")
                self.logger.info(f"Time passed: {time_passed}")
                self.logger.info(f"Time scale: {time_Scale}")
                self.logger.info(f"Time scale * times1[-1]: {time_Scale*times1[-1]}")
                self.logger.info(f"Time scale * times1[-1]: {time_Scale*times1[-1]}")
                self.logger.info(f"Time scale * times1[-1]: {time_Scale*times1[-1]}")
                self.logger.info(f"Time scale * times1[-1]: {time_Scale*times1[-1]}")
                self.logger.info(f"Time scale * times1[-1]: {time_Scale*times1[-1]}")
                self.logger.info(f"Time scale * times1[-1]: {time_Scale*times1[-1]}")
                self.logger.info(f"Time scale * times1[-1]: {time_Scale*times1[-1]}")
                self.logger.info(f"Time scale * times1[-1]: {time_Scale*times1[-1]}")
                self.go_to_pose(0, 0, 1)