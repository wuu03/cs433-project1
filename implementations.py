import numpy as np


def _logistic_sigmoid(t):
    """Apply sigmoid on t."""
    return 1.0 / (1.0 + np.exp(-t))


def _logistic_loss(y, tx, w):
    """Compute the cost by negative log likelihood."""
    pred = _logistic_sigmoid(tx @ w)
    sample_losses = -(y * np.log(pred) + (1.0 - y) * np.log(1.0 - pred))
    loss = np.mean(sample_losses)

    return loss


def _logistic_gradient(y, tx, w):
    """Compute the gradient of loss."""
    pred = _logistic_sigmoid(tx @ w)
    error = pred - y
    num_samples = y.shape[0]
    gradient = (tx.T @ error) / num_samples

    return gradient

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Train logistic regression using gradient descent.

    Args:
        y: Binary labels, shape (N,), with values in {0, 1}.
        tx: Preprocessed feature matrix, shape (N, D).
        initial_w: Initial weights, shape (D,).
        max_iters: Number of gradient descent updates.
        gamma: Learning rate.

    Returns:
        w: Final weights, shape (D,).
        loss: Loss at the final weights.
    """
    w = initial_w.copy()

    for _ in range(max_iters):
        gradient = _logistic_gradient(y, tx, w)
        w = w - gamma * gradient

    # Compute the loss using the final weights.
    loss = _logistic_loss(y, tx, w)

    return w, loss


def reg_logistic_regression(
    y, tx, lambda_, initial_w, max_iters, gamma
):
    """Train L2-regularized logistic regression using gradient descent.

    Args:
        y: Binary labels, shape (N,), with values in {0, 1}.
        tx: Preprocessed feature matrix, shape (N, D).
        lambda_: Nonnegative regularization coefficient.
        initial_w: Initial weights, shape (D,).
        max_iters: Number of gradient descent updates.
        gamma: Learning rate.

    Returns:
        w: Final weights, shape (D,).
        loss: Final loss, excluding the penalty.
    """
    w = initial_w.copy()

    for _ in range(max_iters):
        gradient = _logistic_gradient(y, tx, w)
        gradient = gradient + 2.0 * lambda_ * w
        w = w - gamma * gradient

    loss = _logistic_loss(y, tx, w)

    return w, loss