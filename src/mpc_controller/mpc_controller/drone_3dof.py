from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos

def export_drone_3dof_model() -> AcadosModel:

    model_name = 'drone_3dof'

    # constants
    m = 1.0
    g = 9.81 
    Ix = 1.657171e-5
    Iy = Ix#1.6655602e-5
    Iz = 2.9261652e-5
    d = 0.046
    c = 0.00089468 * 10 # TODO why without this *10 does this oscillate???

    # set up states & controls   
    roll = SX.sym('roll')
    pitch = SX.sym('pitch')
    yaw = SX.sym('yaw')
    droll = SX.sym('droll')
    dpitch = SX.sym('dpitch')
    dyaw = SX.sym('dyaw')

    x = vertcat(roll, pitch, yaw, droll, dpitch, dyaw)

    F1 = SX.sym('F1')
    F2 = SX.sym('F2')
    F3 = SX.sym('F3')
    F4 = SX.sym('F4')
    u = vertcat(F1, F2, F3, F4)

    # xdot
    roll_dot = SX.sym('roll_dot')
    pitch_dot = SX.sym('pitch_dot')
    yaw_dot = SX.sym('yaw_dot')
    droll_dot = SX.sym('droll_dot')
    dpitch_dot = SX.sym('dpitch_dot')
    dyaw_dot = SX.sym('dyaw_dot')

    xdot = vertcat(roll_dot, pitch_dot, yaw_dot, droll_dot, dpitch_dot, dyaw_dot)

    # dynamics
    f_expl = vertcat(droll,
                     dpitch,
                     dyaw,
                     (d / Ix) * (F1 + F4 - F2 - F3),
                     (d / Iy) * (F3 + F4 - F1 - F2),
                     (c / Iz) * (F2 + F4 - F1 - F3)
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
    model.x_labels = [r'roll', r'pitch', r'yaw', r'$\dot{roll}$', r'$\dot{pitch}$', r'$\dot{yaw}$']
    model.u_labels = ['$F1$', '$F2$', '$F3$', '$F4$']
    model.t_label = '$t$ [s]'

    return model
