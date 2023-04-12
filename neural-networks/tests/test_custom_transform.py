import numpy as np
import src.random
from sklearn.linear_model import LogisticRegression


def test_custom_transform():
    from src import load_data
    from src import custom_transform

    X, y, _ = load_data("data/spiral.csv")

    new_X = custom_transform(X)
    msg = "Only use at most three features"
    assert new_X.shape[1] <= 3, msg

    model = LogisticRegression()
    model.fit(new_X, y)
    preds = model.predict(new_X)
    acc = np.mean(preds == y)
    msg = f"Custom transform at {100 * acc:.1f}% accuracy, want 90%."
    assert acc >= 0.9, msg

    # Don't just memorize the order of the data!
    src.random.rng.seed()
    for _ in range(4):
        shuffle = np.argsort(src.random.rand(X.shape[0]))
        X1 = X[shuffle, :]
        y1 = y[shuffle]
        model = LogisticRegression()
        new_X1 = custom_transform(X1)
        model.fit(new_X1, y1)
        preds = model.predict(new_X1)
        acc2 = np.mean(preds == y1)
        msg = "Shuffling the data shouldn't change your accuracy"
        assert np.abs(acc - acc2) < 0.05, msg

    # Don't just memorize the location of every point!
    src.random.rng.seed()
    for _ in range(4):
        X2 = X1 + src.random.normal(0, 0.01, size=X1.shape)
        model = LogisticRegression()
        new_X2 = custom_transform(X2)
        model.fit(new_X2, y1)
        preds = model.predict(new_X2)
        acc3 = np.mean(preds == y1)
        msg = "A tiny bit of random noise shouldn't hurt"
        assert np.abs(acc - acc3) < 0.1, msg
