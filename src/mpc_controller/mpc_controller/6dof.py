import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Imu
import numpy as np
import time
from acados_template import AcadosOcp, AcadosOcpSolver
# from mpc_controller.drone_6dof import export_drone_6dof_model
from mpc_controller.drone_6dof_v2 import export_drone_6dof_model
import casadi as ca

class MPCControlNode(Node):
    def __init__(self):
        super().__init__('mpc_control_node')
        
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0
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
        
        self.dt = 0.01
        self.timer = self.create_timer(self.dt, self.control_update)
        
        self.last_time = 0.0
        self.prev_roll = 0.0
        self.prev_pitch = 0.0   
        self.prev_yaw = 0.0
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.prev_z = 0.0
        
        ### MPC Controller init
        self.ocp = AcadosOcp()
        
        self.model = export_drone_6dof_model()
        self.ocp.model = self.model

        Tf = 1.0
        N = 100

        # set prediction horizon
        self.ocp.solver_options.N_horizon = N
        self.ocp.solver_options.tf = Tf

        # cost matrices
        Q_mat = 2*np.diag([1e2, 1e2, 1e2, 1e0, 1e0, 1e0, 1e1, 1e1, 1e1, 1e0, 1e0, 1e0])
        R_mat = 2*np.diag([1e1, 1e1, 1e1, 1e1])

        # path cost
        self.ocp.cost.cost_type = 'NONLINEAR_LS'
        self.ocp.model.cost_y_expr = ca.vertcat(self.model.x, self.model.u)
        self.ocp.cost.yref = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
        self.ocp.cost.W = ca.diagcat(Q_mat, R_mat).full()

        # terminal cost
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
        self.ocp.cost.yref_e = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        self.ocp.model.cost_y_expr_e = self.model.x
        self.ocp.cost.W_e = Q_mat

        # set constraints
        Fmax = 1
        self.ocp.constraints.lbu = np.zeros(4) 
        self.ocp.constraints.ubu = np.full(4, Fmax)
        self.ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        self.ocp.constraints.x0 = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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
        
        self.dx = (self.x - self.prev_x) / (time.time() - self.last_time)
        self.dy = (self.y - self.prev_y) / (time.time() - self.last_time)
        self.dz = (self.z - self.prev_z) / (time.time() - self.last_time)
        
        self.prev_x, self.prev_y, self.prev_z = self.x, self.y, self.z
        
        self.last_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1000000000.
        
    def imu_callback(self, msg):
        q = [
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z
        ]
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
            
            state = np.array([self.x, self.y, self.z, self.dx, self.dy, self.dz, self.roll, self.pitch, self.yaw, self.droll, self.dpitch, self.dyaw])
            
            # target_yaw = np.arctan2(self.target[1] - self.y, self.target[0] - self.x)
            # target_pitch = np.arctan2(self.target[2] - self.z, np.sqrt((self.target[0] - self.x)**2 + (self.target[1] - self.y)**2))
            
            # yref = np.array([0.0, -target_pitch, target_yaw, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            # yref = np.array([0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            solver = self.ocp_solver
            
            N = self.ocp.solver_options.N_horizon
            
            # for i in range(N):
            #     solver.set(i, 'yref', yref)
    
            solver.set(0, "lbx", state)
            solver.set(0, "ubx", state)
            
            status = solver.solve()
            
            if status != 0:
                # raise Exception(f'acados returned status {status}.')
                print('Acados returned status {status}.')

            # print(solver.get(0, "u"))
            print(solver.get_stats('time_tot'))
            
            u = solver.get(0, "u") #/ 2
            
            motor_commands = Float32MultiArray()
            motor_commands.data = [float(u[0]), float(u[1]), float(u[2]), float(u[3])]
            self.motor_pub.publish(motor_commands)
        
def main():
    rclpy.init()
    node = MPCControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()