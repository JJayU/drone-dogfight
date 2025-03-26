from acados_template import AcadosOcp, AcadosOcpSolver
# from drone_6dof import export_drone_6dof_model
from drone_6dof_v2 import export_drone_6dof_model
import numpy as np
import casadi as ca
from utils import plot_pendulum

def main():
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_drone_6dof_model()
    ocp.model = model

    Tf = 1.0
    nx = model.x.rows()
    nu = model.u.rows()
    N = 100

    # set prediction horizon
    ocp.solver_options.N_horizon = N
    ocp.solver_options.tf = Tf

    # cost matrices
    Q_mat = 2*np.diag([1e2, 1e2, 1e2, 1e0, 1e0, 1e0, 1e1, 1e1, 1e1, 1e0, 1e0, 1e0])
    R_mat = 2*np.diag([1e1, 1e1, 1e1, 1e1])

    # path cost
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
    ocp.cost.yref = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ocp.cost.W = ca.diagcat(Q_mat, R_mat).full()

    # terminal cost
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.yref_e = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.W_e = Q_mat

    # set constraints
    Fmax = 1
    ocp.constraints.lbu = np.zeros(4) 
    ocp.constraints.ubu = np.full(4, Fmax)
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    ocp.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING' # turns on globalization

    ocp_solver = AcadosOcpSolver(ocp)

    simX = np.zeros((N+1, nx))
    simU = np.zeros((N, nu))

    status = ocp_solver.solve()

    if status != 0:
        raise Exception(f'acados returned status {status}.')

    # get solution
    for i in range(N):
        simX[i,:] = ocp_solver.get(i, "x")
        simU[i,:] = ocp_solver.get(i, "u")
    simX[N,:] = ocp_solver.get(N, "x")
    
    # print(simU)

    plot_pendulum(np.linspace(0, Tf, N+1), Fmax, simU, simX, latexify=True, time_label=model.t_label, x_labels=model.x_labels, u_labels=model.u_labels)


if __name__ == '__main__':
    main()