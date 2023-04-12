import numpy as np

from src.data import load_data
from src.model import Model 
from src.loss import BinaryCrossEntropyLoss
from src.layers import FullyConnected, SigmoidActivation


def test_model_forward():
    # xor dataset
    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=float)

    layer1 = FullyConnected(2, 2)
    layer1.weights = np.arange(6, dtype=float).reshape(*layer1.weights.shape)

    model = Model([layer1], None)
    pred = model.forward(X)
    target = np.array([[6, 9], [2, 4], [4, 6], [0, 1]])
    assert np.all(pred == target)

    layer2 = FullyConnected(2, 1)
    layer2.weights = np.arange(3, dtype=float).reshape(*layer2.weights.shape)

    model = Model([layer1, SigmoidActivation(), layer2], None)
    pred = model.forward(X)
    target = np.array([[2.99728059], [2.84482466], [2.97706854], [1.96211716]])
    assert np.all(np.isclose(pred, target))


def test_model_backward():
    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    layer1 = FullyConnected(2, 2)
    w1 = np.arange(6, dtype=float).reshape(*layer1.weights.shape) 
    layer1.weights = w1.copy()

    layer2 = FullyConnected(2, 1)
    w2 = np.arange(3, dtype=float).reshape(*layer2.weights.shape)
    layer2.weights = w2.copy()

    layers = [layer1, SigmoidActivation(), layer2, SigmoidActivation()]

    model = Model(layers, BinaryCrossEntropyLoss(), learning_rate=0.1)

    # This should raise an exception because we haven't been able to
    #     save anything to the `self.input_` of each 
    try:
        model.backward(0.5 * np.ones_like(y), y)
        works = True
    except TypeError as e:
        works = False
    except AttributeError as e:
        works = False
    assert not works, "Can't backprop without computing forward pass"

    pred = model.forward(X)
    loss_before = model.loss.forward(pred, y)

    model.backward(pred, y)
    pred = model.forward(X)
    loss_after = model.loss.forward(pred, y)

    # Loss should go down after a single backward() step
    assert loss_after < loss_before, "Loss didn't decrease"

    # given these initialized weights and this data,
    #     updates should go have these signs
    w1_update = np.sign(w1 - layer1.weights)
    assert np.all(w1_update == np.array([[1, 1], [-1, -1], [1, -1]]))
    w2_update = np.sign(w2 - layer2.weights)
    assert np.all(w2_update == np.array([1, 1, 1]))


def test_model_fit_xor():
    from sklearn.metrics import accuracy_score

    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    hidden_layer_size = 8
    layer1 = FullyConnected(2, hidden_layer_size)
    layer2 = FullyConnected(hidden_layer_size, 1)

    layers = [layer1, SigmoidActivation(), layer2, SigmoidActivation()]

    model = Model(layers, BinaryCrossEntropyLoss(), learning_rate=1)

    msg = "Should get 50% accuracy before training"
    pred_before = model.forward(X)
    acc_before = accuracy_score(pred_before > 0.5, y)
    loss_before = model.loss.forward(pred_before, y)
    assert acc_before == 0.5, msg

    msg = "If you train for 0 iters, nothing should change"
    layer1_weights = layer1.weights.copy()
    model.fit(X, y, max_iter=0)
    assert np.allclose(layer1_weights, layer1.weights), msg

    # Should get 75% accuracy or better after 250 epochs
    max_iter = 200
    model.fit(X, y, max_iter=max_iter)
    pred_middle = model.forward(X)
    acc_middle = accuracy_score(pred_middle > 0.5, y)
    loss_middle = model.loss.forward(pred_middle, y)
    msg = f"Expect {100 * acc_middle:.1f} > 50% after {max_iter}"
    assert acc_middle > 0.5, msg

    # Should get 100% accuracy after 1000 epochs
    max_iter = 1000
    model.fit(X, y, max_iter=max_iter)
    pred_after = model.forward(X)
    acc_after = accuracy_score(pred_after > 0.5, y)
    loss_after = model.loss.forward(pred_after, y)
    msg = f"Expect {100 * acc_middle:.1f} = 100% after {max_iter}"
    assert acc_after == 1.0, msg

    # Re-initialize model
    layer1 = FullyConnected(2, hidden_layer_size)
    layer2 = FullyConnected(hidden_layer_size, 1)
    layers = [layer1, SigmoidActivation(), layer2, SigmoidActivation()]

    # shouldn't be able to learn with too large of learning rate
    model = Model(layers, BinaryCrossEntropyLoss(), learning_rate=100)
    model.fit(X, y, 2 * max_iter)
    acc_bad = accuracy_score(model.predict(X) > 0.5, y)
    assert acc_bad == 0.5, "With large learning rate, should do poorly"


def test_model_fit_circles():
    from sklearn.metrics import accuracy_score

    X, y, _ = load_data("data/circles.csv")

    hidden_layer_size = 8
    layer1 = FullyConnected(2, hidden_layer_size)
    layer2 = FullyConnected(hidden_layer_size, 1)

    layers = [layer1, layer2, SigmoidActivation()]

    learning_rate = 1
    max_iter = 2000

    model = Model(layers, BinaryCrossEntropyLoss(), learning_rate=learning_rate)

    # Without nonlinearity, should get ~85% accuracy
    model.fit(X, y, max_iter)
    accuracy_before = accuracy_score(model.predict(X) > 0.5, y)
    assert accuracy_before < 0.6

    layer1 = FullyConnected(2, hidden_layer_size)
    layer2 = FullyConnected(hidden_layer_size, 1)
    layers = [layer1, SigmoidActivation(), layer2, SigmoidActivation()]
    model = Model(layers, BinaryCrossEntropyLoss(), learning_rate=learning_rate)

    # With nonlinearity, should get 100% accuracy
    model.fit(X, y, max_iter)
    accuracy_after = accuracy_score(model.predict(X) > 0.5, y)
    assert accuracy_after == 1.0
