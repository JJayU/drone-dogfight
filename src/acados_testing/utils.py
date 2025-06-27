import matplotlib.pyplot as plt
import numpy as np
from acados_template import latexify_plot
import matplotlib

matplotlib.use('Qt5Agg')

def plot_pendulum(t, u_max, U, X_true, latexify=False, plt_show=True, time_label='$t$', x_labels=None, u_labels=None):
    """
    Params:
        t: time values of the discretization
        u_max: maximum absolute value of u
        U: array with shape (N_sim-1, nu) or (N_sim, nu)
        X_true: array with shape (N_sim, nx)
        latexify: latex style plots
    """

    if latexify:
        latexify_plot()

    nx = X_true.shape[1] - 1
    nu = U.shape[1] if U.ndim > 1 else 1  # Number of control inputs

    fig, axes = plt.subplots(nx + nu, 1, sharex=True)

    for i in range(nx):
        axes[i].plot(t, X_true[:, i])
        axes[i].grid()
        if x_labels is not None:
            axes[i].set_ylabel(x_labels[i])
        else:
            axes[i].set_ylabel(f'$x_{i}$')

    # Plot each control input on a separate subplot
    U = np.atleast_2d(U)  # Ensure U is 2D
    for j in range(nu):
        ax = axes[nx + j]
        ax.step(t, np.append([U[0, j]], U[:, j]))
        ax.set_ylabel(f'$u_{j}$' if u_labels is None else u_labels[j])
        ax.hlines(u_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
        ax.hlines(-u_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
        ax.set_ylim([-1.2 * u_max, 1.2 * u_max])
        ax.grid()
    
    axes[-1].set_xlim(t[0], t[-1])
    axes[-1].set_xlabel(time_label)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)
    fig.align_ylabels()

    if plt_show:
        plt.show()
