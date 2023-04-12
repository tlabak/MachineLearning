import numpy as np

from src.data import load_data
from src.model import Model 
from src.loss import SquaredLoss
from src.layers import FullyConnected, SigmoidActivation


def test_squared_loss_basics():
    from sklearn.metrics import mean_squared_error

    A = np.arange(1, 20).reshape(-1, 1)
    B = np.arange(25, 6, -1).reshape(-1, 1)

    loss = SquaredLoss()

    # Hint: this should be a mean across all 19 "rows"
    assert loss.forward(A, B) == mean_squared_error(A, B)

    # gradient should be an array of shape [19, 1]
    grad = np.arange(-48, 25, 4).reshape(-1, 1)
    assert np.all(loss.backward() == grad)


def test_squared_loss_fit():
    from sklearn.metrics import mean_squared_error

    X, y, _ = load_data("data/polynomial_regression.csv")
    layers = [
        FullyConnected(1, 4),
        SigmoidActivation(),
        FullyConnected(4, 1),
    ]

    # for test case, set weights explicitly
    w1 = np.array([
        [0.49671415, -0.1382643, 0.64768854, 1.52302986],
        [-0.2341533, -0.2341369, 1.57921282, 0.76743473]
    ])
    w2 = np.array([
        [0.10021694], [0.11342611], [-0.2971744], [0.47718901], [0.58224571]
    ])
    layers[0].weights = w1.copy()
    layers[2].weights = w2.copy()

    # do a first step backwards
    model = Model(layers, SquaredLoss(), 0.2)
    pred_before = model.forward(X)
    loss_before = model.loss.forward(pred_before, y)
    model.backward(pred_before, y)

    # first step should move weights in these directions
    w1_diff = np.sign(layers[0].weights - w1)
    assert np.all(w1_diff == np.array([[-1, 1, -1, -1], [-1, 1, -1, -1]]))
    w2_diff = np.sign(layers[2].weights - w2)
    assert np.all(w2_diff == -1 * np.ones_like(w2))

    # first step should decrease the loss
    pred_middle = model.forward(X)
    loss_middle = model.loss.forward(pred_middle, y)
    assert loss_middle < loss_before
    
    # If we train the model for 10k steps, should get loss around ~0.07
    model.fit(X, y, max_iter=10000)
    loss_after = model.loss.forward(model.forward(X), y)
    assert loss_after < 0.1
