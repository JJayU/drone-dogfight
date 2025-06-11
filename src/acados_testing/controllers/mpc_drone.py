from controllers.interface import Controller
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from mpc_controller.drone_6dof_v2 import export_drone_6dof_model
import numpy as np
import casadi as ca

# MPC Controller for a drone in a MuJoCo simulation.

def compute_u(self, state, yref):
    
    solver = self.ocp_solver
    
    print(state[:3])
    
    solver.set(0, "lbx", state)
    solver.set(0, "ubx", state)
    
    for i in range(self.N):
        solver.set(i, "yref", yref)
        
    solver.set(self.N, "yref", yref[:12])
    
    status = solver.solve()
    
    if status != 0:
        raise Exception(f'acados returned status {status}.')

    # print(solver.get(0, "u"))
    print(solver.get_stats('time_tot'))
    
    return solver.get(0, "u")


class MPCController(Controller):
    def __init__(self, env):
        super().__init__(env)
        
        self.ocp = AcadosOcp()
        
        self.model = export_drone_6dof_model()
        self.ocp.model = self.model

        Tf = 1.0
        self.N = 200

        # set prediction horizon
        self.ocp.solver_options.N_horizon = self.N
        self.ocp.solver_options.tf = Tf

        # cost matrices
        Q_mat = 2*np.diag([1e1, 1e1, 1e2, 1e0, 1e0, 1e0, 1e2, 1e2, 1e2, 1e0, 1e0, 1e0])
        R_mat = 2*np.diag([1e1, 1e1, 1e1, 1e1])

        # path cost
        self.ocp.cost.cost_type = 'NONLINEAR_LS'
        self.ocp.model.cost_y_expr = ca.vertcat(self.model.x, self.model.u)
        #                              x    y    z    dx   dy   dz   r    p    y    dr   dp   dy   m1   m2   m3   m4
        self.ocp.cost.yref = np.array([5.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.8, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
        self.ocp.cost.W = ca.diagcat(Q_mat, R_mat).full()

        # terminal cost
        self.ocp.cost.cost_type_e = 'NONLINEAR_LS'
        #                                x    y    z    dx   dy   dz   r    p    y    dr   dp   dy
        self.ocp.cost.yref_e = np.array([5.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.8, 0.0, 0.0, 0.0])
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
        self.ocp.solver_options.integrator_type = 'ERK'
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI, SQP
        self.ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization
        
        self.ocp_solver = AcadosOcpSolver(self.ocp)


    def compute_control(self, yref):

        return [compute_u(self, self.env._get_observation(), yref)]
