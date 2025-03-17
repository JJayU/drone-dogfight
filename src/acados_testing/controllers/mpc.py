from controllers.interface import Controller
import numpy as np
import gymnasium as gym
from acados_template import AcadosOcp, AcadosOcpSolver
from pendulum import export_pendulum_model
import numpy as np
import casadi as ca

def compute_u(self, state):
    
    theta = state[0] + np.pi # Conversion (0.0 is down in Acados and up in Gym)
    dtheta = state[1]
    
    solver = self.ocp_solver
    
    solver.set(0, "lbx", np.array([theta, dtheta]))
    solver.set(0, "ubx", np.array([theta, dtheta]))
    
    status = solver.solve()
    
    if status != 0:
        raise Exception(f'acados returned status {status}.')

    print(theta)
    print(dtheta)
    print(solver.get(0, "u"))
    print(solver.get_stats('time_tot'))
    
    return solver.get(0, "u")


class MPCController(Controller):
    def __init__(self, env):
        super().__init__(env)
        
        self.ocp = AcadosOcp()
        
        self.model = export_pendulum_model()
        self.ocp.model = self.model

        Tf = 3.0
        N = 200

        # set prediction horizon
        self.ocp.solver_options.N_horizon = N
        self.ocp.solver_options.tf = Tf

        # cost matrices
        Q_mat = 2*np.diag([1e4, 1e-3])
        R_mat = 2*np.diag([1e1])

        # path cost
        self.ocp.cost.cost_type = 'NONLINEAR_LS'
        self.ocp.model.cost_y_expr = ca.vertcat(self.model.x, self.model.u)
        self.ocp.cost.yref = np.array([np.pi, 0.0, 0.0])
        self.ocp.cost.W = ca.diagcat(Q_mat, R_mat).full()

        # terminal cost
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
        self.ocp.cost.yref_e = np.array([np.pi, 0.0])
        self.ocp.model.cost_y_expr_e = self.model.x
        self.ocp.cost.W_e = Q_mat

        # set constraints
        Fmax = 2
        self.ocp.constraints.lbu = np.array([-Fmax])
        self.ocp.constraints.ubu = np.array([+Fmax])
        self.ocp.constraints.idxbu = np.array([0])
        
        self.ocp.constraints.x0 = np.array([0.0, 0.0])
        

        # set options
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI, SQP
        
        self.ocp_solver = AcadosOcpSolver(self.ocp)



    def compute_control(self):

        return [compute_u(self, self.env.unwrapped.state)]
