from acados_template import AcadosModel
from casadi import SX, vertcat

def export_drone_6dof_model():
    model_name = 'crazyflie'

    # parameters
    g = 9.8066    # [m/s^2] gravity
    m = 33e-3     # [kg] mass
    Ixx = 1.395e-5
    Iyy = 1.395e-5
    Izz = 2.173e-5
    Cd = 7.9379e-06 # [N/krpm^2] Drag coef
    Ct = 3.25e-4    # [N/krpm^2] Thrust coef
    d = 65e-3 / 2   # [m] half distance between motors

    # states
    x = SX.sym('x')
    y = SX.sym('y')
    z = SX.sym('z')
    q1 = SX.sym('q1')
    q2 = SX.sym('q2')
    q3 = SX.sym('q3')
    q4 = SX.sym('q4')
    dx = SX.sym('dx')
    dy = SX.sym('dy')
    dz = SX.sym('dz')
    droll = SX.sym('droll')
    dpitch = SX.sym('dpitch')
    dyaw = SX.sym('dyaw')
    states = vertcat(x, y, z, q1, q2, q3, q4, dx, dy, dz, droll, dpitch, dyaw)

    # controls
    w1 = SX.sym('w1')
    w2 = SX.sym('w2')
    w3 = SX.sym('w3')
    w4 = SX.sym('w4')
    controls = vertcat(w1, w2, w3, w4)

    # state derivatives
    xdot = SX.sym('xdot', states.shape[0])

    # Forces and torques
    U1 = Ct * (w1**2 + w2**2 + w3**2 + w4**2)
    U2 = Ct * d * (w3**2 + w4**2 - w1**2 - w2**2)
    U3 = Ct * d * (w1**2 + w4**2 - w2**2 - w3**2)
    U4 = Cd * (w2**2 + w4**2 - w1**2 - w3**2)

    # Dynamics
    f_expl = vertcat(
        dx,
        dy,
        dz,
        - (q2 * droll) / 2 - (q3 * dpitch) / 2 - (q4 * dyaw) / 2,
        (q1 * droll) / 2 - (q4 * dpitch) / 2 + (q3 * dyaw) / 2,
        (q4 * droll) / 2 + (q1 * dpitch) / 2 - (q2 * dyaw) / 2,
        (q2 * dpitch) / 2 - (q3 * droll) / 2 + (q1 * dyaw) / 2,
        g * (2 * q1 * q3 - 2 * q2 * q4),
        -g * (2 * q1 * q2 + 2 * q3 * q4),
        U1 / m,
        U2 / Ixx,
        U3 / Iyy,
        U4 / Izz
    )

    f_impl = xdot - f_expl

    # Model definition
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = states
    model.xdot = xdot
    model.u = controls
    model.name = model_name

    # Labels
    model.x_labels = [r'x', r'y', r'z', r'q1', r'q2', r'q3', r'q4', r'dx', r'dy', r'dz', r'droll', r'dpitch', r'dyaw']
    model.u_labels = ['w1', 'w2', 'w3', 'w4']
    model.t_label = 't [s]'

    return model
