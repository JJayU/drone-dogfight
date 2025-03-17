#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos

def export_drone_3dof_model() -> AcadosModel:

    model_name = 'drone_3dof'

    # constants
    m = 1.0# mass of the ball [kg]
    g = 9.81 # gravity constant [m/s^2]
    Ix = 1.395e-5
    Iy = 1.395e-5
    Iz = 2.173e-5
    d = 65e-3 / 2.
    c = 3.25e-4

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
