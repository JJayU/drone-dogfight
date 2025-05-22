import numpy as np
from casadi import SX, vertcat, mtimes, inv
from acados_template import AcadosModel

def export_drone_6dof_model():
    model_name = "drone_6dof"
    
    # Parametry
    g = 9.81  # przyspieszenie grawitacyjne [m/s^2]
    m = 0.028   # masa drona [kg]
    l = 0.046  # ramię quadrotora [m]
    k = 0.00089468 # współczynnik momentu reakcyjnego
    J = SX([[1.657171e-5, 0, 0], [0, 1.657171e-5, 0], [0, 0, 2.9261652e-5]])  # macierz bezwładności
    
    # Zmienne stanu
    px = SX.sym('px')
    py = SX.sym('py')
    pz = SX.sym('pz')
    vx = SX.sym('vx')
    vy = SX.sym('vy')
    vz = SX.sym('vz')
    phi = SX.sym('phi')
    theta = SX.sym('theta')
    psi = SX.sym('psi')
    wx = SX.sym('wx')
    wy = SX.sym('wy')
    wz = SX.sym('wz')
    
    x = vertcat(px, py, pz, vx, vy, vz, phi, theta, psi, wx, wy, wz)
    xdot = SX.sym('xdot', x.size1())
    
    # Wejścia sterujące (ciągi od silników)
    T1 = SX.sym('T1')
    T2 = SX.sym('T2')
    T3 = SX.sym('T3')
    T4 = SX.sym('T4')
    
    u = vertcat(T1, T2, T3, T4)
    
    # Siła nośna
    T = (T1 + T2 + T3 + T4) * 0.15 
    
    # Macierz rotacji
    R = SX.zeros(3, 3)
    R[0, 0] = SX.cos(psi) * SX.cos(theta)
    R[0, 1] = SX.cos(psi) * SX.sin(theta) * SX.sin(phi) - SX.sin(psi) * SX.cos(phi)
    R[0, 2] = SX.cos(psi) * SX.sin(theta) * SX.cos(phi) + SX.sin(psi) * SX.sin(phi)
    R[1, 0] = SX.sin(psi) * SX.cos(theta)
    R[1, 1] = SX.sin(psi) * SX.sin(theta) * SX.sin(phi) + SX.cos(psi) * SX.cos(phi)
    R[1, 2] = SX.sin(psi) * SX.sin(theta) * SX.cos(phi) - SX.cos(psi) * SX.sin(phi)
    R[2, 0] = -SX.sin(theta)
    R[2, 1] = SX.cos(theta) * SX.sin(phi)
    R[2, 2] = SX.cos(theta) * SX.cos(phi)
    
    # Dynamika translacji
    dp = vertcat(vx, vy, vz)
    dv = (1/m) * (mtimes(R, vertcat(0, 0, T)) - vertcat(0, 0, m*g))
    
    # Dynamika rotacji
    tau_x = l * (T1 + T4 - T2 - T3) 
    tau_y = l * (T3 + T4 - T1 - T2) 
    tau_z = k * (T2 + T4 - T1 - T3)
    tau = vertcat(tau_x, tau_y, tau_z)
    
    w = vertcat(wx, wy, wz)
    
    # Macierz skew-symmetric dla wektora prędkości kątowej
    w_skew = SX.zeros(3, 3)
    w_skew[0, 1] = -wz
    w_skew[0, 2] = wy
    w_skew[1, 0] = wz
    w_skew[1, 2] = -wx
    w_skew[2, 0] = -wy
    w_skew[2, 1] = wx
    
    dw = mtimes(inv(J), (tau - mtimes(w_skew, mtimes(J, w))))
    
    # Połączone równania stanu
    x_dot = vertcat(dp, dv, wx, wy, wz, dw[0], dw[1], dw[2])
    
    # Definicja modelu Acados
    model = AcadosModel()
    model.f_expl_expr = x_dot
    model.f_impl_expr = x_dot - xdot  # Poprawiona definicja implikowanego modelu
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name
    model.x_labels = ['px', 'py', 'pz', 'vx', 'vy', 'vz', 'phi', 'theta', 'psi', 'wx', 'wy', 'wz']
    model.u_labels = ['T1', 'T2', 'T3', 'T4']
    
    return model
