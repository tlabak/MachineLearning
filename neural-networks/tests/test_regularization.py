import numpy as np

from src.data import load_data
from src.model import Model 
from src.loss import BinaryCrossEntropyLoss
from src.layers import FullyConnected, SigmoidActivation
from src.regularization import Regularizer


def test_l1_basics():
    k = 20
    X = np.arange(k) * np.power(-1, np.arange(0, k))
    X = X.reshape(5, 4)
    X = np.concatenate([X, np.zeros([1, 4])], axis=0)

    regularizer = Regularizer(penalty="l1", alpha=np.pi)

    grad = regularizer.grad(X)

    # Don't regularize the intercepts
    assert np.all(grad[0, :] == 0)

    # Do regularize all other nonzero values
    assert np.all(grad[1:5, (0, 2)] == np.pi)
    assert np.all(grad[1:5, (1, 3)] == -np.pi)
    
    # Gradient of 0 is zero
    assert np.all(grad[5, :] == 0)


def test_l2_basics():
    k = 20
    X = np.arange(k) * np.power(-1, np.arange(0, k))
    X = X.reshape(5, 4)
    X = np.concatenate([X, np.zeros([1, 4])], axis=0)

    regularizer = Regularizer(penalty="l2", alpha=1)

    grad = regularizer.grad(X)

    # Don't regularize the intercepts
    assert np.all(grad[0, :] == 0)
    # Gradient of 0 is zero
    assert np.all(grad[5, :] == 0)

    # Do regularize all other nonzero values
    assert np.all(grad[1:5, 0] == np.arange(8, 33, 8))
    assert np.all(grad[1:5, 3] == np.arange(-14, -39, -8))


def test_regularization_fit_circles():
    from sklearn.metrics import accuracy_score

    X, y, _ = load_data("data/circles.csv")
    hidden_layer_size = 16

    weights = {}
    alphas = [0, 0.001, 0.01, 0.1]
    for alpha in alphas:
        for penalty in ["l1", "l2"]:

            reg = Regularizer(penalty=penalty, alpha=alpha)

            layers = [
                FullyConnected(2, hidden_layer_size, regularizer=reg),
                SigmoidActivation(),
                FullyConnected(hidden_layer_size, 1, regularizer=reg),
                SigmoidActivation(),
            ]
            model = Model(layers, BinaryCrossEntropyLoss(), 0.1)
            model.fit(X, y, 100)

            weights[(alpha, penalty)] = layers[0].weights[1:, :]

    prev_mean = np.inf
    for alpha in alphas:
        # For L2 regularization, we should see decreasing
        #     magnitudes of our weights
        mean = np.mean(np.square(weights[(alpha, "l2")]))
        assert mean < prev_mean, f"L2 reg with alpha={alpha}"
        prev_mean = mean

    baseline = np.mean(np.square(weights[(0, "l1")]))
    for alpha in alphas[1:]:
        # For L1 regularization, should still decrease
        #   compared to unregularized data
        mean = np.mean(np.square(weights[(alpha, "l1")]))
        assert mean < baseline, f"L1 reg with alpha={alpha}"

    # L1 with strong alpha should create many near-zero weights
    assert np.mean(np.abs(weights[(0.1, "l1")]) < 0.01) > 0.7

    # L1 should have some larger values
    l1_max = np.max(np.abs(weights[(0.1, "l1")]))
    l2_max = np.max(np.abs(weights[(0.1, "l2")]))
    assert l1_max > l2_max

    # L1 should have more near-zero weights
    l1_near_zero = np.mean(np.abs(weights[(0.1, "l1")]) < 0.01)
    l2_near_zero = np.mean(np.abs(weights[(0.1, "l2")]) < 0.01)
    assert l1_near_zero > l2_near_zero
