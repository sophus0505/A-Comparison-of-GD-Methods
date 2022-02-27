import numpy as np
import matplotlib.pyplot as plt

# The Rosenbrock function and its gradient with b = 100


def Rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def Grad_Rosenbrock(x, y):
    f1 = 2 * (200 * x ** 3 - 200 * x * y + x - 1)
    f2 = 200 * (y - x ** 2)

    return np.array([f1, f2])

# The Rosenbrock funcition and its gradient with b = 1


def Rosenbrock1(x, y):
    return (1 - x) ** 2 + 1 * (y - x ** 2) ** 2


def Grad_Rosenbrock1(x, y):
    g1 = 2 * (2 * x ** 3 - 2 * x * y + x - 1)
    g2 = 2 * (y - x ** 2)

    return np.array([g1, g2])


def gradient_descent(F, dF, x0, y0, stepsize, epsilon=1e-7, max_iters=10000):
    """Perform SGD on a function F : R2 -> R with gradient dF.

    Args:
        F (function): A function F : R2 -> R
        dF (function): The gradient of F
        x0 (float): Initial position in x-direction
        y0 (float): Initial position in y-direction
        stepsize (float): The learning rate
        epsilon (float, optional): Margin of error. Defaults to 1e-7.
        max_iters (int, optional): Maximum allowed iterations. Defaults to 10000.

    Returns:
        vals (List[ndarray]): List of all the points visited during SGD
        i (int): The number of iterations used before convergence
    """
    uncertainty = float("inf")

    point = np.array([x0, y0])

    vals = [point]

    # simple check if the sequence has begun to diverge
    # (might want to change value if working with other functions that Rosenbrock)
    threshold = 1e10
    i = 0
    # Iterate over the sequence
    while uncertainty > epsilon and i < max_iters:
        point_prev = point
        point = point - stepsize * dF(*point)

        if point[0] > threshold or point[1] > threshold:
            return "Divergent", float("inf")

        uncertainty = np.linalg.norm(point - point_prev)
        vals.append(point)
        i += 1

    return vals, i


def armijo(F, dF, point, alpha, sigma):
    """Checks if Armijo's condition holds.

    Args:
        F (function): A function F : R2 -> R
        dF (function): The gradient of F
        point (ndarray): The current point in the algorithm
        alpha (float): Parameter
        sigma (float): Current learning rate

    Returns:
        (Bool): True if Armijo's condition holds, else False
    """
    grad = dF(*point)
    if sigma >= 100:
        return False
    left = F(*(point - sigma*grad)) - F(*point)
    right = -alpha*sigma*(grad[0] * grad[0] + grad[1] * grad[1])
    return left <= right


def backtracking_gradient_descent(
    F, dF, x0, y0, alpha=0.5, beta=0.5, delta0=1, epsilon=1e-7, max_iters=10_000
):
    """Performs BGD on F.

    Args:
        F (function): A function F : R2 -> R
        dF (function): The gradient of F
        x0 (float): Initial position in the x-direction
        y0 (float): Initial position in the y-direction
        alpha (float, optional): Parameter for Armijo's condition. Defaults to 0.5.
        beta (float, optional): Parameter used to change lr. Defaults to 0.5.
        delta0 (float, optional): Initial learning rate. Defaults to 1.
        epsilon (float, optional): Margin of error. Defaults to 1e-10.
        max_iters (int, optional): Maximum allowed iterations. Defaults to 1000.

    Returns:
        vals (List[ndarray]): List of all the points visited during SGD
        i (int): The number of iterations used before convergence
    """

    uncertainty = float('inf')

    point = np.array([x0, y0],  dtype=np.float64)
    vals = [point]

    sigma = delta0

    threshold = 1e15
    i = 0

    while uncertainty > epsilon and i < max_iters:

        if point[0] > threshold or point[1] > threshold:
            return "Divergent", float("inf")

        # Armijo's condition, sigma becomes the stepsize
        while not armijo(F, dF, point, alpha, sigma) and sigma > epsilon:
            sigma = beta * sigma

        point_prev = point
        point = point - sigma * dF(*point)

        uncertainty = np.linalg.norm(point - point_prev)
        vals.append(point)

        i += 1

    return vals, i


def two_way_backtracking_gradient_descent(
    F, dF, x0, y0, alpha=0.5, beta=0.5, delta0=1, epsilon=1e-10, max_iters=1000
):
    """Performs two-way BGD on F.

    Args:
        F (function): A function F : R2 -> R
        dF (function): The gradient of F
        x0 (float): Initial position in the x-direction
        y0 (float): Initial position in the y-direction
        alpha (float, optional): Parameter for Armijo's condition. Defaults to 0.5.
        beta (float, optional): Parameter used to change lr. Defaults to 0.5.
        delta0 (float, optional): Initial learning rate. Defaults to 1.
        epsilon (float, optional): Margin of error. Defaults to 1e-10.
        max_iters (int, optional): Maximum allowed iterations. Defaults to 1000.

    Returns:
        vals (List[ndarray]): List of all the points visited during SGD
        i (int): The number of iterations used before convergence
    """

    uncertainty = float("inf")

    point = np.array([x0, y0])
    vals = [point]

    sigma = delta0

    i = 0

    while uncertainty > epsilon and i < max_iters:

        # Armijo's condition, sigma becomes the stepsize
        while not armijo(F, dF, point, alpha, sigma) and sigma > epsilon:
            sigma = beta * sigma

        while armijo(F, dF, point, alpha, sigma) and sigma > epsilon:
            sigma = sigma / beta

        point_prev = point
        point = point - sigma * dF(*point)

        uncertainty = np.linalg.norm(point - point_prev)
        vals.append(point)

        i += 1

    return vals, i


if __name__ == "__main__":
    pass
