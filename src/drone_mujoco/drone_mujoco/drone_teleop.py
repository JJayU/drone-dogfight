import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray
import numpy as np
from scipy.spatial.transform import Rotation
import math
from tf_transformations import euler_from_quaternion

class DroneController(Node):
    def __init__(self):
        super().__init__('drone_controller')

        self.current_thrust = 0.0
        self.current_motors = 0.0

        self.Kp_z = 1.5 
        self.Ki_z = 0.1
        self.Kd_z = 1.0 

        self.Kp_yaw = .05 
        self.Ki_yaw = 0.00001
        self.Kd_yaw = .001
        
        self.min_thrust = 0.0
        self.max_thrust = 1.0
        
        self.target_z= 1.0  
        self.target_x= 1.0
        self.target_y= 1.0  
        self.current_yaw = 0.0

        self.current_x = 0.0 
        self.current_y = 0.0 
        self.current_z = 0.0 

        self.prev_error = 0.0
        self.integral = 0.0

        self.prev_error_yaw = 0.0
        self.integral_yaw = 0.0

        self.pub_height = self.create_publisher(Float32MultiArray,'motor_power',10)
        self.create_timer(0.02,self.loop)

        self.sub_gps = self.create_subscription(PointStamped, 'gps', self.gps_callback, 10)

        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)

        target_pos = {

        }

    def imu_callback(self, msg: Imu):
        orientation = msg.orientation
        _, _, yaw = euler_from_quaternion([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ])
        self.current_yaw = yaw

    def gps_callback(self, msg):
        """ Aktualizacja wysokości z danych GPS. """
        self.current_z = msg.point.z
        self.curent_x = msg.point.x
        self.curent_y = msg.point.y

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def loop(self):
        """ Główna pętla sterowania wysokością. """
        error = self.target_z - self.current_z
        self.integral += error * 0.02
        derivative = (error - self.prev_error) / 0.02
        self.prev_error = error

        cmd_roll = 0.0
        cmd_pitch = 0.0
        cmd_yaw = 0.0
        cmd_thrust = 0.06

        target_yaw = math.atan2(self.target_y - self.current_y,
                        self.target_x - self.current_x)

        yaw_error = self.normalize_angle(target_yaw - self.current_yaw)

        self.integral_yaw += yaw_error  * 0.02
        derivative_yaw = (yaw_error - self.prev_error_yaw) / 0.02
        self.prev_error_yaw = yaw_error

        # Oblicz moc silnika na podstawie PID
        thrust_z = ((self.Kp_z * error) + (self.Ki_z * self.integral) + (self.Kd_z * derivative))/10
        cmd_yaw = ((self.Kp_yaw * yaw_error) + (self.Ki_yaw * self.integral_yaw) + (self.Kd_yaw * derivative_yaw))/10

        # Ograniczenie wartości mocy
        thrust = max(self.min_thrust, min(self.max_thrust, thrust_z))

        motorPower_m1 = thrust - cmd_roll + cmd_pitch - cmd_yaw
        motorPower_m2 = thrust - cmd_roll - cmd_pitch + cmd_yaw
        motorPower_m3 = thrust + cmd_roll - cmd_pitch - cmd_yaw
        motorPower_m4 = thrust + cmd_roll + cmd_pitch + cmd_yaw

        motors = np.clip([motorPower_m1, motorPower_m2, motorPower_m3, motorPower_m4], self.min_thrust, self.max_thrust)

        self.current_thrust = thrust
        self.current_motors = motors

        # Publikowanie wartości do topicu ROS2
        msg = Float32MultiArray()
        msg.data = motors.tolist()
        self.pub_height.publish(msg)

        self.get_logger().info(f'Current height: {self.current_z}, {target_yaw},{self.current_yaw}')

def main():
    rclpy.init()
    controller = DroneController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()