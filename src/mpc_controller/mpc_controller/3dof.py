import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Imu
import numpy as np
import time
from acados_template import AcadosOcp, AcadosOcpSolver
from mpc_controller.drone_3dof import export_drone_3dof_model
import casadi as ca

class MPCControlNode(Node):
    def __init__(self):
        super().__init__('mpc_control_node')
        
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.droll = 0.0
        self.dpitch = 0.0
        self.dyaw = 0.0
        
        self.gps_sub = self.create_subscription(
            PointStamped,
            'gps',
            self.gps_callback,
            10
        )
        self.imu_sub = self.create_subscription(
            Imu,
            'imu',
            self.imu_callback,
            10
        )
        self.target_sub = self.create_subscription(
            PointStamped,
            'target_point',
            self.target_callback,
            10
        )
        
        self.motor_pub = self.create_publisher(
            Float32MultiArray,
            'motor_power',
            10
        )
        
        self.dt = 0.005
        self.timer = self.create_timer(self.dt, self.control_update)
        
        self.last_time = 0.0
        self.prev_roll = 0.0
        self.prev_pitch = 0.0   
        self.prev_yaw = 0.0
        
        ### MPC Controller init
        self.ocp = AcadosOcp()
        
        self.model = export_drone_3dof_model()
        self.ocp.model = self.model

        Tf = 1.0
        N = 200

        # set prediction horizon
        self.ocp.solver_options.N_horizon = N
        self.ocp.solver_options.tf = Tf

        # cost matrices
        Q_mat = 2*np.diag([1e4, 1e4, 1e4, 1e1, 1e1, 1e1])
        R_mat = 2*np.diag([1e2, 1e2, 1e2, 1e2])

        # path cost
        self.ocp.cost.cost_type = 'NONLINEAR_LS'
        self.ocp.model.cost_y_expr = ca.vertcat(self.model.x, self.model.u)
        self.ocp.cost.yref = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.ocp.cost.W = ca.diagcat(Q_mat, R_mat).full()

        # terminal cost
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
        self.ocp.cost.yref_e = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
        self.ocp.model.cost_y_expr_e = self.model.x
        self.ocp.cost.W_e = Q_mat

        # set constraints
        Fmax = 1
        self.ocp.constraints.lbu = np.zeros(4) 
        self.ocp.constraints.ubu = np.full(4, Fmax)
        self.ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        self.ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # set options
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
        self.ocp.solver_options.integrator_type = 'IRK'
        self.ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
        self.ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization

        self.ocp_solver = AcadosOcpSolver(self.ocp)
        
    def gps_callback(self, msg):
        self.x = msg.point.x
        self.y = msg.point.y
        self.z = msg.point.z
        
        self.last_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1000000000.
        
    def imu_callback(self, msg):
        q = [
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z
        ]
        # print(msg)
        self.roll, self.pitch, self.yaw = self.quaternion_to_euler(q)
        
        self.droll = (self.roll - self.prev_roll) / (time.time() - self.last_time)
        self.dpitch = (self.pitch - self.prev_pitch) / (time.time() - self.last_time)
        self.dyaw = (self.yaw - self.prev_yaw) / (time.time() - self.last_time)
        
        self.prev_roll, self.prev_pitch, self.prev_yaw = self.roll, self.pitch, self.yaw
        
    def target_callback(self, msg):
        self.target = [msg.point.x, msg.point.y, msg.point.z]
        
    def quaternion_to_euler(self, q):
        w, x, y, z = q
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)
        
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
        
    def control_update(self):
        
        if(time.time() - self.last_time < 1):
            
            state = np.array([self.roll, self.pitch, self.yaw, self.droll, self.dpitch, self.dyaw])
            
            solver = self.ocp_solver
    
            solver.set(0, "lbx", state)
            solver.set(0, "ubx", state)
            
            status = solver.solve()
            
            if status != 0:
                # raise Exception(f'acados returned status {status}.')
                print('acados returned status {status}.')

            # print(solver.get(0, "u"))
            # print(solver.get_stats('time_tot'))
            
            u = solver.get(0, "u")
            
            u = u
            
            motor_commands = Float32MultiArray()
            motor_commands.data = [float(u[0]), float(u[1]), float(u[2]), float(u[3])]
            # motor_commands.data = [0.0001, 0.0001, 0.00, 0.00]
            self.motor_pub.publish(motor_commands)
            
            print(self.pitch)
            # print(self.dpitch)
            
            # print(state)
            
            # self.angle += 0.001
            
            # target_x = np.sin(self.angle)
            # target_y = np.cos(self.angle)
            
            # self.x_pos_pid.setpoint = target_x
            # self.y_pos_pid.setpoint = target_y 
            
            # target_yaw = np.arctan2(self.target[1] - self.y, self.target[0] - self.x)
            
            # self.height_pid.setpoint = self.target[2] - 0.02
        
            # desired_pitch = self.x_pos_pid.update(self.x, self.dt)
            # desired_roll = self.y_pos_pid.update(self.y, self.dt)

            # desired_pitch = np.clip(desired_pitch, -0.5, 0.5) 
            # desired_roll = np.clip(desired_roll, -0.5, 0.5) 
            
            # self.pitch_pid.setpoint = desired_pitch * np.cos(self.yaw) - desired_roll * np.sin(-self.yaw)
            # self.roll_pid.setpoint = - desired_roll * np.cos(-self.yaw) + desired_pitch * np.sin(self.yaw)
            # self.yaw_pid.setpoint = target_yaw
            
            # height_control = self.height_pid.update(self.z, self.dt)
            # pitch_control = self.pitch_pid.update(self.pitch, self.dt)
            # roll_control = self.roll_pid.update(self.roll, self.dt)
            # yaw_control = self.yaw_pid.update(self.yaw, self.dt)
            
            # m1 = height_control - pitch_control + roll_control - yaw_control
            # m2 = height_control - pitch_control - roll_control + yaw_control
            # m3 = height_control + pitch_control - roll_control - yaw_control
            # m4 = height_control + pitch_control + roll_control + yaw_control
            
            # max_thrust = max(abs(m1), abs(m2), abs(m3), abs(m4))
            # if max_thrust > 1.0:
            #     m1 /= max_thrust
            #     m2 /= max_thrust
            #     m3 /= max_thrust
            #     m4 /= max_thrust
            
            # motor_commands = Float32MultiArray()
            # motor_commands.data = [float(m1), float(m2), float(m3), float(m4)]
            # self.motor_pub.publish(motor_commands)
            
            # print('hello')
        
def main():
    rclpy.init()
    node = MPCControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()