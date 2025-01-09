import rclpy
from rclpy.time import Time
from nav_msgs.msg import Odometry

from math import degrees
from scipy.spatial.transform import Rotation as R
import numpy as np

# from choirbot import Pose
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist

import cffirmware as cffirmware # crazyflie firmware

# Software in the loop motor controller
class SILMotorCtrl: 
    def init(self, webots_node, properties):
        
        # Declare the robot name and fix the timestep
        self.__robot    = webots_node.robot
        self.timestep   = int(self.__robot.getBasicTimeStep())
        self.robot_name = self.__robot.getName()

        # Initialize webots driver node
        rclpy.init(args=None)
        self.namespace = str(self.robot_name)
        self.cf_sil_driver = rclpy.create_node(
                            'cf_sil_driver',
                            namespace=self.namespace,
                            allow_undeclared_parameters=True,
                            automatically_declare_parameters_from_overrides=True)

        ## Initialize motors
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
        
        ## Initialize Sensors
        self.__gps = self.__robot.getDevice('gps')
        self.__gyro = self.__robot.getDevice('gyro')
        self.__imu = self.__robot.getDevice('inertial_unit')
        self.__gps.enable(self.timestep)
        self.__gyro.enable(self.timestep)
        self.__imu.enable(self.timestep)
        
        ## Intialize Variables
        self.current_pose = Pose()
        self.current_twist = Twist()
        self.past_position = np.zeros(3)
        self.past_time = self.__robot.getTime()

        # Initialize cf classes
        self.state      = cffirmware.state_t()
        self.sensors    = cffirmware.sensorData_t()
        self.setpoint   = cffirmware.setpoint_t()
        self.control    = cffirmware.control_t()

        # Initialize flags
        self.initialization = True
        self.emergency_stop = False

        # Safe zone limits
        self.safezone_limits = [np.infty,np.infty,np.infty] # [x,y,z] in meters

    def step(self):
        raise NotImplementedError

    def init_setpoint(self):
        raise NotImplementedError

    def setpoint_callback(self):
        raise NotImplementedError

    def compute_setpoint(self):
        raise NotImplementedError

    def stop(self,_):
        self.emergency_stop = True

    def check_safety_area(self):
        safezone_ck = [
            np.abs(self.current_pose.position.x) > self.safezone_limits[0],
            np.abs(self.current_pose.position.y) > self.safezone_limits[1],
            np.abs(self.current_pose.position.z) > self.safezone_limits[2]
        ]
        if any(safezone_ck):
            self.emergency_stop = True
            self.cf_sil_driver.get_logger().warn('{} KILLED: OUT OF SAFE ZONE'.format(self.robot_name))        
        

    def update_current_pose(self):
        ## Get measurements
        self.current_rpy                = np.array(self.__imu.getRollPitchYaw())
        quatarnian                      = np.array(R.from_euler('xyz', self.current_rpy).as_quat())   
        gyro_rad_s                      = np.array(self.__gyro.getValues())
        gps_position                    = np.array(self.__gps.getValues())
        gps_speed                       = np.array(self.__gps.getSpeedVector())
        
        self.current_pose.orientation.x = quatarnian[0]
        self.current_pose.orientation.y = quatarnian[1]
        self.current_pose.orientation.z = quatarnian[2]
        self.current_pose.orientation.w = quatarnian[3]
        
        self.current_twist.angular.x    = gyro_rad_s[0]
        self.current_twist.angular.y    = gyro_rad_s[1]
        self.current_twist.angular.z    = gyro_rad_s[2]
        
        self.current_pose.position.x    = gps_position[0]
        self.current_pose.position.y    = gps_position[1]
        self.current_pose.position.z    = gps_position[2]

        self.current_twist.linear.x     = gps_speed[0]
        self.current_twist.linear.y     = gps_speed[1]
        self.current_twist.linear.z     = gps_speed[2]

    def publish_odometry(self):
        odom = Odometry()
        odom.header.stamp = Time(seconds=self.__robot.getTime()).to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.current_pose.position.x
        odom.pose.pose.position.y = self.current_pose.position.y
        odom.pose.pose.position.z = self.current_pose.position.z

        odom.twist.twist.linear.x = self.current_twist.linear.x
        odom.twist.twist.linear.y = self.current_twist.linear.y
        odom.twist.twist.linear.z = self.current_twist.linear.z

        odom.pose.pose.orientation.x = self.current_pose.orientation.x
        odom.pose.pose.orientation.y = self.current_pose.orientation.y
        odom.pose.pose.orientation.z = self.current_pose.orientation.z
        odom.pose.pose.orientation.w = self.current_pose.orientation.w

        odom.twist.twist.angular.x = self.current_twist.angular.x
        odom.twist.twist.angular.y = self.current_twist.angular.y
        odom.twist.twist.angular.z = self.current_twist.angular.z
        self.odom_publisher.publish(odom)

    def update_cf_state(self):
        self.state.attitude.roll     = degrees(self.current_rpy[0])
        self.state.attitude.pitch    = -degrees(self.current_rpy[1])
        self.state.attitude.yaw      = degrees(self.current_rpy[2])
        self.state.position.x        = self.current_pose.position.x
        self.state.position.y        = self.current_pose.position.y
        self.state.position.z        = self.current_pose.position.z
        self.state.velocity.x        = self.current_twist.linear.x
        self.state.velocity.y        = self.current_twist.linear.y
        self.state.velocity.z        = self.current_twist.linear.z

        debug_string = f"state    -> pos x: {self.state.position.x:.4f}\ty: {self.state.position.y:.4f}\tz: {self.state.position.z:.4f}\n            vel x: {self.state.velocity.x:.4f}\ty: {self.state.velocity.y:.4f}\tz: {self.state.velocity.z:.4f}"
        # print(string)

    def update_cf_sensors(self):
        self.sensors.gyro.x = degrees(self.current_twist.angular.x)
        self.sensors.gyro.y = degrees(self.current_twist.angular.y)
        self.sensors.gyro.z = degrees(self.current_twist.angular.z)
        
    def udpate_past_time(self):
        self.past_time = self.get_time()
        
    def get_time(self):
        return self.__robot.getTime()

    def send_motor_cmd(self, cmd_thrust, cmd_roll, cmd_pitch, cmd_yaw):
        ## Motor mixing
        # Power distribution from ~/crazyflie-firmware/src/modules/src/power_distribution_quadrotor.c

        motorPower_m1 =  cmd_thrust - cmd_roll + cmd_pitch + cmd_yaw
        motorPower_m2 =  cmd_thrust - cmd_roll - cmd_pitch - cmd_yaw
        motorPower_m3 =  cmd_thrust + cmd_roll - cmd_pitch + cmd_yaw
        motorPower_m4 =  cmd_thrust + cmd_roll + cmd_pitch - cmd_yaw

        scaling = 800 #700 # 1000 ##Todo, remove necessity of this scaling (SI units in firmware)
        self.m1_motor.setVelocity(-motorPower_m1/scaling)
        self.m2_motor.setVelocity(motorPower_m2/scaling)
        self.m3_motor.setVelocity(-motorPower_m3/scaling)
        self.m4_motor.setVelocity(motorPower_m4/scaling)
