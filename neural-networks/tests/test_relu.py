import numpy as np

from src.data import load_data
from src.model import Model 
from src.loss import BinaryCrossEntropyLoss
from src.layers import FullyConnected, SigmoidActivation, ReluActivation


def test_relu_basics():
    k = 20
    X = np.arange(k) * np.power(-1, np.arange(0, k))
    X = X.reshape(5, 4)

    relu = ReluActivation()

    # ReLU should zero out negatives in the forward pass
    forward = relu.forward(X)
    assert np.all(forward >= 0)
    assert np.sum(forward[:, (1, 3)]) == 0

    # Use the saved values to zero out gradients
    backward = relu.backward(np.ones([5, 1]))
    assert np.all(backward >= 0)
    assert np.sum(backward[:, (1, 3)]) == 0

    # Which means negative gradients stay negative!
    backward = relu.backward(-1 * np.ones([5, 1]))
    assert np.all(backward <= 0)
    assert np.sum(backward[:, (1, 3)]) == 0


def test_relu_fit_xor():
    from sklearn.metrics import accuracy_score

    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    hidden_layer_size = 8
    layer1 = FullyConnected(2, hidden_layer_size)
    layer2 = FullyConnected(hidden_layer_size, 1)

    layers = [layer1, ReluActivation(), layer2, SigmoidActivation()]
    model = Model(layers, BinaryCrossEntropyLoss(), learning_rate=0.1)

    # Should get 50% accuracy before training
    pred_before = model.forward(X)
    acc_before = accuracy_score(pred_before > 0.5, y)
    loss_before = model.loss.forward(pred_before, y)
    assert acc_before == 0.5

    # Should get 100% accuracy after 1000 epochs
    model.fit(X, y, max_iter=1000)
    pred_after = model.forward(X)
    acc_after = accuracy_score(pred_after > 0.5, y)
    loss_after = model.loss.forward(pred_after, y)
    assert acc_after == 1.0


def test_relu_fit_circles():
    from sklearn.metrics import accuracy_score

    X, y, _ = load_data("data/circles.csv")

    hidden_layer_size = 8
    layer1 = FullyConnected(2, hidden_layer_size)
    layer2 = FullyConnected(hidden_layer_size, 1)
    layers = [layer1, ReluActivation(), layer2, SigmoidActivation()]

    model = Model(layers, BinaryCrossEntropyLoss(), learning_rate=0.1)

    # With relu, should get >90% accuracy
    model.fit(X, y, 2000)
    accuracy_after = accuracy_score(model.predict(X) > 0.5, y)
    assert accuracy_after >= 0.90
