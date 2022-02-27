import pylab
import matplotlib
import time

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numdifftools as nd
import params as pr
import matplotlib.colors as colors


from mpl_toolkits import mplot3d

from algorithms import (
    Rosenbrock,
    Grad_Rosenbrock,
    Rosenbrock1,
    Grad_Rosenbrock1,
    gradient_descent,
    backtracking_gradient_descent,
    two_way_backtracking_gradient_descent,
    armijo,
)

matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "beramono",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)

plt.rcParams.update(
    {
        "font.family": "beramono",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
    }
)


def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


plt.rcParams["figure.figsize"] = set_size(345)


def plot_contour(F, dF, x0, y0, alpha, beta, delta0, stepsize, epsilon):
    """Generates a contour plot that compares how SGD, BGD and 2W-BGD converges
    to a local minimum in F.

    Args:
        F (function): A function F : R2 -> R
        dF (function): The gradient of F
        x0 (float): Starting point on x-axis
        y0 (float): Starting point on y-axis
        alpha (float): alpha value, for BGD and 2W-BGD
        beta (float): beta value, for BGD and 2W-BGD
        delta0 (float): Initial lr for BGD and 2W-BGD
        stepsize (float): Lr for SGD
        epsilon (float): Minimum size of sigma, and uncertainty
    """
    palette = sns.color_palette("inferno")

    x = np.linspace(-2, 2)
    y = np.linspace(-1, 3)
    X, Y = np.meshgrid(x, y)
    Z = Rosenbrock(X, Y)

    # Initialize the figure and plot the contour of the Rosenbrock func
    plt.figure(figsize=set_size(345))
    plt.contourf(X, Y, Z, 50, levels=20, cmap="viridis")
    plt.xlabel('$x$')
    plt.ylabel('$y$')

    # Generate data (perform methods)
    vals, i = gradient_descent(F, dF, x0, y0, stepsize, epsilon)
    vals2, i2 = backtracking_gradient_descent(
        F, dF, x0, y0, alpha, beta, delta0, epsilon)
    vals3, i3 = two_way_backtracking_gradient_descent(
        F, dF, x0, y0, alpha, beta, delta0, epsilon)

    # print(i, i2, i3)
    if vals2 == "Divergent":
        print("The sequence diverges")
        return

    plt.scatter(*zip(*vals), label="GD", s=10, color=palette[2])
    plt.scatter(*zip(*vals2), label="BGD", s=10, color=palette[5])
    plt.scatter(*zip(*vals3), label="2W-BGD", s=10)

    # Plot the global minimum
    plt.scatter(1, 1, marker="+", color="w", label="Global minimum")

    plt.plot(*zip(*vals), alpha=0.5, color=palette[2])
    plt.plot(*zip(*vals2), alpha=0.5, color=palette[5])
    plt.plot(*zip(*vals3), alpha=0.5)

    plt.legend()
    plt.savefig("../figures/contour_plot3.pgf")
    plt.show()


def calc_iterations(F, dF, x0, y0, alpha, beta, epsilon, max_iterations):
    """Calculates the number of iterations used to converge to a local minimum of F by
    SGD, BGD and two-way BGD for initial learning rates [100, 10, 1, 0.1, 0.01, 0.001, 0.0001].
    The results are printed out in the terminal. 

    Args:
        F (function): A function F: R2 -> R
        dF (function): The gradient of F
        x0 (float): Initial position on x-axis
        y0 (float): Initial position on y-axis
        alpha (float): Parameter used for Armijo's condition
        beta (float): Parameter for BGD and two-way BGD
        epsilon (float): Margin of error
        max_iterations (int): The maximum amount of iterations allowed
    """

    np.seterr(all='raise')

    for stepsize in [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]:

        start1 = time.time()
        _, gd = gradient_descent(
            F,
            dF,
            x0,
            y0,
            stepsize,
            epsilon=epsilon,
            max_iters=max_iterations,
        )
        end1 = time.time()
        time1 = end1 - start1

        start2 = time.time()
        __, bgd = backtracking_gradient_descent(
            F,
            dF,
            x0,
            y0,
            alpha=alpha,
            beta=beta,
            delta0=stepsize,
            epsilon=epsilon,
            max_iters=max_iterations,
        )
        end2 = time.time()
        time2 = end2 - start2

        start3 = time.time()
        ___, twobgd = two_way_backtracking_gradient_descent(
            Rosenbrock,
            Grad_Rosenbrock,
            x0,
            y0,
            alpha=alpha,
            beta=beta,
            delta0=stepsize,
            epsilon=epsilon,
            max_iters=max_iterations,
        )
        end3 = time.time()
        time3 = end3 - start3

        print(f"{stepsize = }")
        print(f"{gd          = }, {time1    = }")
        print(f"{bgd         = }, {time2    = }")
        print(f"{twobgd      = }, {time3    = }")
        print("######################")


if __name__ == "__main__":

    # Establish a common style for the plots
    palette = sns.color_palette("inferno")
    sns.set_palette("inferno")
    # plt.style.use("ggplot")

    x0, y0 = -1, 2

    # Nice parameters:
    plot_contour(Rosenbrock1, Grad_Rosenbrock1, x0, y0,
                 alpha=0.5, beta=0.6, delta0=1, stepsize=0.1, epsilon=1e-7)

    # calc_iterations(
    #     Rosenbrock, Grad_Rosenbrock, x0, y0, alpha=0.5, beta=0.5, epsilon=1e-7, max_iterations=500_000
    # )

    plt.show()
