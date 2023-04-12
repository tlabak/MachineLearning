import numpy as np
import random, string
import csv
from .test_utils import write_random_csv_file


def test_load_data():
    from src import load_data

    n_features = np.random.randint(5, 20)
    n_samples = np.random.randint(50, 150)
    features, targets, attribute_names = write_random_csv_file(n_features, n_samples)

    _features, _targets, _attribute_names = load_data('tests/test.csv')
    assert attribute_names == _attribute_names
    assert np.allclose(features, _features) and np.allclose(targets.flatten(), _targets.flatten())


def test_train_test_split():
    from src import load_data
    from src import train_test_split

    n_features = np.random.randint(5, 20)
    n_samples = np.random.randint(50, 150)
    features, targets, attribute_names = write_random_csv_file(n_features, n_samples)
    fraction = np.random.rand()

    output = train_test_split(features, targets, fraction)
    expected_train_size = int(n_samples * fraction)
    expected_test_size = n_samples - expected_train_size

    for o in output:
        assert o.shape[0] == expected_train_size or o.shape[0] == expected_test_size

    # Remember, your train_test_split should be deterministic
    # You shouldn't be randomly shuffling the data!
    features = np.stack([np.arange(n_samples) for _ in range(n_features)], axis=1)
    targets = np.ones([n_samples])
    for fraction in np.arange(0.1, 1.0, 0.1).tolist():
        X_train, _, _, _ = train_test_split(features, targets, fraction)
        assert np.isclose(np.mean(X_train) * 2 + 1, int(fraction * n_samples))
    
    features, targets, _ = load_data("data/blobs.csv")
    _, y_train, _, y_test = train_test_split(features, targets, 0.5)
    assert np.isclose(np.mean(y_train), 0.5079365079365079), np.mean(y_train)
    assert np.isclose(np.mean(y_test), 0.359375), np.mean(y_test)



def test_cross_validation():
    from src import load_data
    from src import cross_validation

    # Remember, your train_test_split should be deterministic
    # You shouldn't be randomly shuffling the data!
    n = 10
    features = np.arange(n).reshape(-1, 1)
    targets = np.arange(n).reshape(-1, 1)
    for folds in [2, 5, 10]:
        cv = cross_validation(features, targets, folds)
        for i, arr in enumerate(cv):
            X_train, _, X_test, _ = arr
            assert X_train.shape[0] == (folds - 1) * X_test.shape[0]
            assert X_test[0] == n * i // folds

    features, targets, _ = load_data("data/noisy.csv")
    cv = cross_validation(features, targets, 5)
    means = []
    for (_, y_train, _, y_test) in cv:
        means.append((np.mean(y_train), np.mean(y_test)))
    targets = np.array(
        [[0.525,  0.65],
         [0.5375, 0.6],
         [0.55,   0.55],
         [0.5625, 0.5],
         [0.575,  0.45]])
    assert np.all(np.isclose(np.array(means), targets))
