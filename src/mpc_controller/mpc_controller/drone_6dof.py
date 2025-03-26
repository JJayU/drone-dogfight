from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, sqrt

def export_drone_6dof_model() -> AcadosModel:

    model_name = 'drone_6dof'

    # constants
    m = 0.028
    g = 9.81 
    Ix = 1.657171e-5
    Iy = Ix#1.6655602e-5
    Iz = 2.9261652e-5
    d = 0.046
    c = 0.00089468 *10 # TODO why without this *10 does this oscillate???
    k_p = 0.15

    # set up states & controls   
    px = SX.sym('x')
    py = SX.sym('y')
    pz = SX.sym('z')
    dpx = SX.sym('dx')
    dpy = SX.sym('dy')
    dpz = SX.sym('dz')
    roll = SX.sym('roll')
    pitch = SX.sym('pitch')
    yaw = SX.sym('yaw')
    droll = SX.sym('droll')
    dpitch = SX.sym('dpitch')
    dyaw = SX.sym('dyaw')

    x = vertcat(px, py, pz, dpx, dpy, dpz, roll, pitch, yaw, droll, dpitch, dyaw)

    F1 = SX.sym('F1')
    F2 = SX.sym('F2')
    F3 = SX.sym('F3')
    F4 = SX.sym('F4')
    u = vertcat(F1, F2, F3, F4)

    # xdot
    px_dot = SX.sym('px_dot')
    py_dot = SX.sym('py_dot')
    pz_dot = SX.sym('pz_dot')
    dpx_dot = SX.sym('dpx_dot')
    dpy_dot = SX.sym('dpy_dot')
    dpz_dot = SX.sym('dpz_dot')
    roll_dot = SX.sym('roll_dot')
    pitch_dot = SX.sym('pitch_dot')
    yaw_dot = SX.sym('yaw_dot')
    droll_dot = SX.sym('droll_dot')
    dpitch_dot = SX.sym('dpitch_dot')
    dyaw_dot = SX.sym('dyaw_dot')

    xdot = vertcat(px_dot, py_dot, pz_dot, dpx_dot, dpy_dot, dpz_dot, roll_dot, pitch_dot, yaw_dot, droll_dot, dpitch_dot, dyaw_dot)
    
    U1 = (F1 + F2 + F3 + F4) * 0.15 - m * g 
    U2 = (d / sqrt(2)) * (F1 + F3 - F2 - F4)             
    U3 = (d / sqrt(2)) * (F1 + F2 - F3 - F4)               
    U4 = c * (F2 + F4 - F1 - F3)
    
    # dynamics
    f_expl = vertcat(dpx,
                     dpy,
                     dpz,
                     g * pitch,
                     g * roll,
                     U1 / m,
                     droll,
                     dpitch,
                     dyaw,
                     U2 / Ix,
                     U3 / Iy,
                     U4 / Iz
    )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    # store meta information
    model.x_labels = [r'x', r'y', r'z', r'$\dot{x}$', r'$\dot{y}$', r'$\dot{z}$' r'roll', r'pitch', r'yaw', r'$\dot{roll}$', r'$\dot{pitch}$', r'$\dot{yaw}$']
    model.u_labels = ['$F1$', '$F2$', '$F3$', '$F4$']
    model.t_label = '$t$ [s]'

    return model
