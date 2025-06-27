import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PointStamped, PoseStamped
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from mpc_controller.drone_6dof_v2 import export_drone_6dof_model
import casadi as ca

# This is a ROS2 node that implements an MPC controller for a drone using the acados library.

class MPCControlNode(Node):
    def __init__(self):
        super().__init__('mpc_control_node')
        
        ### Subscribers
        self.state_sub = self.create_subscription(
            Float32MultiArray,
            'full_state',
            self.state_callback,
            10
        )
        
        self.target_sub = self.create_subscription(
            PointStamped,
            'target_point',
            self.target_callback,
            10
        )
        
        self.drone_des_pos_sub = self.create_subscription(
            PoseStamped,
            "drone_des_pos",
            self.drone_des_pos_callback,
            10
        )
        
        self.motor_pub = self.create_publisher(
            Float32MultiArray,
            'motor_power',
            10
        )
        
        ### Control loop setup
        self.dt = 0.02
        self.timer = self.create_timer(self.dt, self.control_update)
        
        ### Global variables
        self.target = [0.0, 0.0, 0.0]
        self.full_state = np.zeros(12)
        
        ### MPC Controller init
        self.ocp = AcadosOcp()
        
        self.model = export_drone_6dof_model()
        self.ocp.model = self.model

        Tf = 1.0
        N = 50
        
        self.tempi = 1000
        self.x = 0.0
        self.y = 0.0
        
        self.drone_des_pos = [0.0, 0.0, 0.0]
        self.drone_des_orientation = [0.0, 0.0, 0.0]

        # set prediction horizon
        self.ocp.solver_options.N_horizon = N
        self.ocp.solver_options.tf = Tf

        # cost matrices
        Q_mat = 2*np.diag([1e1, 1e1, 1e2, 1e0, 1e0, 1e0, 1e2, 1e2, 1e3, 1e0, 1e0, 1e0])
        R_mat = 2*np.diag([1e1, 1e1, 1e1, 1e1])

        # path cost
        self.ocp.cost.cost_type = 'NONLINEAR_LS'
        self.ocp.model.cost_y_expr = ca.vertcat(self.model.x, self.model.u)
        #                              x    y    z    dx   dy   dz   r    p    y    dr   dp   dy   m1   m2   m3   m4
        self.ocp.cost.yref = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
        self.ocp.cost.W = ca.diagcat(Q_mat, R_mat).full()

        # terminal cost
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
        #                                x    y    z    dx   dy   dz   r    p    y    dr   dp   dy
        self.ocp.cost.yref_e = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
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
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI, SQP
        self.ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization

        self.ocp_solver = AcadosOcpSolver(self.ocp)
        
        
    def state_callback(self, msg):
        self.full_state = np.array(msg.data.tolist())
        
        
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
    
    def drone_des_pos_callback(self, msg):
        self.drone_des_pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        r,p,y = self.quaternion_to_euler([
            msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])
        self.drone_des_orientation = [r, p, y]
        
    def angle_difference(self, target, current):
        """
        Returns the shortest difference between two angles (in radians), in range [-pi, pi].
        """
        diff = target - current
        return (diff + np.pi) % (2 * np.pi) - np.pi    
    
    def control_update(self):
        solver = self.ocp_solver

        # Aktualny yaw drona
        current_yaw = self.full_state[8]
        # Zadany yaw
        target_yaw = self.drone_des_orientation[2]
        # Najkrótsza różnica kątowa
        yaw_ref = current_yaw + self.angle_difference(target_yaw, current_yaw)

        # Update reference state
        yref = np.array([
            self.drone_des_pos[0], self.drone_des_pos[1], self.drone_des_pos[2],
            0.0, 0.0, 0.0,
            0.0, 0.0, yaw_ref,
            0.0, 0.0, 0.0,
            0.5, 0.5, 0.5, 0.5
        ])

        for i in range(self.ocp.solver_options.N_horizon):
            solver.set(i, 'yref', yref)
        solver.set(self.ocp.solver_options.N_horizon, 'yref', yref[:12])

        # Update current state
        solver.set(0, "lbx", self.full_state)
        solver.set(0, "ubx", self.full_state)

        status = solver.solve()

        if status != 0:
            raise Exception(f'acados returned status {status}.')

        # Get computation time
        print(solver.get_stats('time_tot'))

        u = solver.get(0, "u")

        # Publish motor commands
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