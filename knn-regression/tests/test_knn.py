import numpy as np
import os
import json

from src import load_data, KNearestNeighbor, generate_regression_data
from src.random import rng


datasets = {}
for fn in os.listdir("data"):
    if ("csv" in fn) and ("frq" not in fn):
        datasets[fn] = os.path.join("data", fn)

def run(data_path, fraction, **kwargs):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # load data into numpy array, and split data
    X, Y, attributes = load_data(data_path)
    if fraction == 1:
        X_train, Y_train = X, Y
        X_test, Y_test = X, Y
    else:
        np.random.seed(0)
        X_train, X_test, Y_train, Y_test = train_test_split(
              X, Y, test_size=fraction, random_state=0, shuffle=False)

    # instantiate model with using KNN kwargs, then train and test
    model = KNearestNeighbor(
        kwargs.get("n_neighbors", 1),
        distance_measure=kwargs.get("distance_measure", "euclidean"),
        aggregator=kwargs.get("aggregator", "mode")
    )
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)

    return accuracy_score(Y_test, pred)


def test_knn_aggregators():
    X = np.ones([5, 1])
    y = np.array([1, 1, 3, 4, 6]).reshape(-1, 1)

    mode_aggregator = KNearestNeighbor(n_neighbors=5, aggregator="mode")
    mode_aggregator.fit(X, y)
    msg = "KNN mode aggregator should take mode of neighbors"
    assert np.all(mode_aggregator.predict(X) == 1), msg

    mean_aggregator = KNearestNeighbor(n_neighbors=5, aggregator="mean")
    mean_aggregator.fit(X, y)
    msg = "KNN mean aggregator should take mean of neighbors"
    assert np.all(mean_aggregator.predict(X) == 3), msg


def test_knn_k_is_1():
    goals = {
      'ivy-league.csv': {1.0: 0.9},
      'majority-rule.csv': {1.0: 0.9, 0.5: 0.6},
      'circles.csv': {1.0: 1.0, 0.5: 0.8},
      'blobs.csv': {1.0: 1.0, 0.5: 1.0},
    }

    learner_type = 'knn'
    for key in goals:
        for fraction, goal in goals[key].items():
            for distance_measure in ["euclidean", "manhattan", "cosine"]:
                accuracy = run(datasets[key], fraction,
                               n_neighbors=1,
                               distance_measure=distance_measure,
                               aggregator="mode")
                message = f"On {key} with fraction={fraction}, distance={distance_measure[:3]},"
                message += f" expected {goal:.1f} but got {accuracy:.3f}"
                assert accuracy >= goal, message


def test_knn_k_is_big():
    from src import load_data
    for key, data_path in datasets.items():
        features, targets, _ = load_data(data_path)
        mode_target = np.argmax(np.unique(targets, return_counts=True)[1])

        model = KNearestNeighbor(
            n_neighbors=features.shape[0],
            distance_measure="euclidean",
            aggregator="mode")
        model.fit(features, targets)
        preds = model.predict(features)

        assert np.all(preds == mode_target), "For large K, KNN just predicts majority class"


def test_knn_comparisons():
    comparisons = [
        # cosine > euclidean for MovieLens data
        ('movielens.csv.gz', 0.5,
         {'distance_measure': 'cosine', 'n_neighbors': 7, "aggregator": "mode"},
         {'distance_measure': 'euclidean', 'n_neighbors': 7, "aggregator": "mode"}),

        # euclidean > cosine for parallel lines
        ('parallel-lines.csv', 0.5,
         {'distance_measure': 'euclidean', 'n_neighbors': 3, "aggregator": "mode"},
         {'distance_measure': 'cosine', 'n_neighbors': 3, "aggregator": "mode"}),
    ]

    for i, comparison in enumerate(comparisons):
        (key, fraction, kwargs_a, kwargs_b) = comparison

        data_path = datasets[key]
        acc_a = run(data_path, fraction, **kwargs_a)
        acc_b = run(data_path, fraction, **kwargs_b)

        message = f"{kwargs_a} should beat {kwargs_b} in comparion #{i + 1}"
        message += f" on {key}, but {acc_a:.3f} <= {acc_b:.3f}"
        assert acc_a > acc_b, message


def test_knn_regression():
    from sklearn.metrics import mean_squared_error

    degrees = range(1, 9)
    amounts = [10, 100]
    k_values = [1, 3, 5]

    for degree in degrees:
        prev_k_mses = np.zeros([len(amounts)])
        for k in k_values:
            new_k_mses = []
            model = KNearestNeighbor(k, distance_measure="euclidean", aggregator="mean")
            prev_amount_mse = np.inf
            for amount in amounts:
                rng.seed()
                x, y = generate_regression_data(degree, amount, amount_of_noise=0.0)
                model.fit(x, y)
                mse = mean_squared_error(y, model.predict(x))
                new_k_mses.append(mse)

                msg0 = f"degree={degree}, amount={amount}, k={k}"
                if k == 1:
                    msg = f"{msg0}: should be perfect"
                    assert np.isclose(mse, 0), msg

                msg = f"{msg0}: {mse:.3f} should beat {prev_amount_mse:.3f}"
                assert mse <= prev_amount_mse, msg
                prev_amount_mse = mse

                msg = f"{msg0}: {mse:.3f} should be <= 0.2"
                if amount == 100:
                    assert mse <= 0.2, msg

                msg = f"{msg0}: {mse:.3f} should be >= 0.05"
                if degree > 4 and k > 3 and amount == 10:
                    assert mse >= 0.05, msg

            new_k_mses = np.array(new_k_mses)
            msg = f"Increasing k to {k} should increase mse, but "
            msg += "{} > {}".format(
                str(np.round(prev_k_mses, 2)),
                str(np.round(new_k_mses, 2)))
            assert np.all(prev_k_mses <= new_k_mses), msg
            prev_k_mses = new_k_mses
