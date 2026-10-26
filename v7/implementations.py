"""The six required estimators. Losses are evaluated at the returned weights.

MSE = mean((y - Xw)**2) / 2; logistic loss is mean cross entropy.
L2 objectives add lambda_ * ||w||**2, but returned losses omit that term.
"""

import numpy as np


def compute_loss(y, tx, w):
    """Return half the mean squared residual."""
    return np.mean((y - tx @ w) ** 2) / 2


def sigmoid(z):
    """Stable logistic sigmoid, including for very large positive/negative inputs."""
    return np.exp(-np.logaddexp(0, -np.asarray(z)))


def logistic_loss(y, tx, w):
    """Mean binary cross entropy computed directly from logits."""
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression with exactly max_iters full gradient steps."""
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        w -= gamma * (tx.T @ (tx @ w - y)) / len(y)
    return w, compute_loss(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression with one uniformly sampled observation per step.

    Call np.random.seed before this function to reproduce the sampled sequence.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        i = np.random.randint(len(y))
        w -= gamma * tx[i] * (tx[i] @ w - y[i])
    return w, compute_loss(y, tx, w)


def _solve(a, b):
    """Solve normal equations, using a pseudoinverse if rank deficient."""
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a) @ b


def least_squares(y, tx):
    """Least squares via normal equations (without numpy.linalg.lstsq)."""
    w = _solve(tx.T @ tx, tx.T @ y)
    return w, compute_loss(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Minimize MSE + lambda_ * ||w||**2; return unpenalized MSE."""
    w = _solve(tx.T @ tx + 2 * len(y) * lambda_ * np.eye(tx.shape[1]), tx.T @ y)
    return w, compute_loss(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Binary logistic regression with y in {0, 1}, using full gradient descent."""
    return reg_logistic_regression(y, tx, 0.0, initial_w, max_iters, gamma)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Full gradient descent for cross entropy + lambda_ * ||w||**2."""
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        gradient = tx.T @ (sigmoid(tx @ w) - y) / len(y) + 2 * lambda_ * w
        w -= gamma * gradient
    return w, logistic_loss(y, tx, w)
