import numpy as np 


def test_perceptron():
    from src import Perceptron, load_data
    from sklearn.metrics import accuracy_score

    features, targets, _ = load_data('data/parallel-lines.csv')
    max_iter = 100
    p = Perceptron(max_iter=max_iter, learning_rate=1.0)
    num_iter_to_converge = p.fit(features, targets)
    targets_hat = p.predict(features)
    assert targets_hat.shape == targets.shape

    msg = "your perceptron should fit this parallel-lines perfectly"
    assert accuracy_score(targets, targets_hat) == 1.0, msg
    assert num_iter_to_converge < max_iter, msg

    p = Perceptron(max_iter=max_iter, learning_rate=0.0)
    num_iter_to_converge = p.fit(features, targets)
    targets_hat = p.predict(features)

    msg = "with learning rate = 0, perceptron can't learn"
    assert accuracy_score(targets, targets_hat) < 1.0, msg
    assert num_iter_to_converge == max_iter, msg


def test_polynomial_perceptron():
    from src import Perceptron, load_data
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import PolynomialFeatures

    features, targets, _ = load_data('data/circles.csv')
    max_iter = 1000
    p = Perceptron(max_iter=max_iter)
    num_iter = p.fit(features, targets)
    targets_hat = p.predict(features)

    msg = "linear perceptron can't fit circles"
    assert accuracy_score(targets, targets_hat) < 1.0, msg

    msg = "after polynomial transform, should fit perfectly"
    poly_features = PolynomialFeatures(2).fit_transform(features)
    num_iter = p.fit(poly_features, targets)
    targets_hat = p.predict(poly_features)
    assert accuracy_score(targets, targets_hat) == 1.0, msg
