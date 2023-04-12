import numpy as np
import os
import warnings

from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from src import Model, FullyConnected
from src import SigmoidActivation
from src import BinaryCrossEntropyLoss


def load_digits(classes=range(10), n=None):
    """
    Helper function to load full MNIST dataset
        The first time this is called, it will take a bit to download 15MB
        of MNIST data to `data/openml/openml.org/data/v1/download/52667.gz`

    classes: which classes to include (all by default)
    n: if not None, limit the data to the first n examples
    """

    downloading = False
    mnist_fn = "data/openml/openml.org/data/v1/download/52667.gz"
    if not os.path.isfile(mnist_fn):
        downloading = True
        message = "\n".join([
            "NOTE: downloading MNIST data; this may take a minute.",
            "      If this does not print \"DONE\" within five minutes,",
            "      see `help_with_mnist.md` to download MNIST from Canvas."])
        print(message)

    X, y = fetch_openml("mnist_784", version=1, data_home="data/",
                        parser="auto", return_X_y=True, as_frame=False)
    if downloading:
        print("DONE")

    y = y.astype(int)

    # Check to make sure that `fetch_openml` is working as expected
    #   If these fail, ask for help.
    assert np.all(y[:10] == np.array([5, 0, 4, 1, 9, 2, 1, 3, 1, 4]))
    assert np.all(np.isclose(
        np.mean(X[:10, :], axis=1),
        np.array([35.10841837, 39.6619898, 24.7997449, 21.85586735,
                  29.60969388, 37.75637755, 22.50765306, 45.74872449,
                  13.86989796, 27.93877551])))

    # Optionally only look at the first n datapoints
    if n is not None:
        X = X[:n, :]
        y = y[:n]

    # Limit the data to `classes` only
    y_mapping = {c: y == c for c in classes}
    where = np.logical_or.reduce(tuple(y_mapping.values()))
    new_y = y.copy()

    # if we only have two classes, should be referenced as 0, 1
    #   even if they are e.g., 7s and 9s.
    for i, cls in enumerate(classes):
        new_y[y_mapping[cls]] = i

    return X[where], new_y[where]


def build_model(hidden_layer_sizes):
    """
    Using the code from `src/`, build a MLP
    """
    layers = []
    input_dim = 784
    for size in hidden_layer_sizes:
        layers.append(FullyConnected(input_dim, size))
        layers.append(SigmoidActivation())
        input_dim = size
    layers.append(FullyConnected(input_dim, 1))
    layers.append(SigmoidActivation())

    return Model(layers, BinaryCrossEntropyLoss(), 0.1)


def two_class():
    """
    Trains our MLP, sklearn's MLP, and sklearn's logistic regression
        on two classes.
    On Zach's M1 Macbook, this takes ~20 seconds to run
    """
    # Consider changing these lines for your answers in Q1a
    X, y = load_digits(classes=(3, 6), n=5000)
    hidden_layer_sizes = (16, )
    max_iter = 200


    # Don't change the rest of this function
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train and evaluate our model
    our_model = build_model(hidden_layer_sizes)
    our_model.fit(X_train, y_train, max_iter=max_iter)
    our_acc = accuracy_score(our_model.predict(X_test) > 0.5, y_test)

    # Train and evaluate sklearn MLP
    sk_model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        sk_model.fit(X_train, y_train)
    sk_acc = accuracy_score(sk_model.predict(X_test), y_test)

    # Train and evaluate sklearn logreg
    logreg = LogisticRegression(max_iter=max_iter)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        logreg.fit(X_train, y_train)
    lr_acc = accuracy_score(logreg.predict(X_test), y_test)

    n_cls = np.unique(y).shape[0]
    print((f"{n_cls}-class| Us: {our_acc:.3f}; Sklearn: {sk_acc:.3f}"
           + f" Logreg: {lr_acc:.3f}"))


def ten_class():
    """
    Trains sklearn's MLP and logistic regression on all ten classes
    On Zach's M1 Macbook, this takes ~45 seconds to run
    """

    # Consider changing these lines for your answers in Q1b
    hidden_layer_sizes = (32, 32, 16, 8)
    max_iter = 400
    sklearn_kwargs = {'alpha': 0.001, 'solver': 'adam', 'learning_rate_init': 0.01, 'beta_1': 0.8, 'beta_2': 0.99}
    X, y = load_digits(classes=range(10), n=10000)

    # Don't change the rest of the function
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train and evaluate sklearn MLP
    sk_model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter,
        **sklearn_kwargs)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        sk_model.fit(X_train, y_train)
    sk_acc = accuracy_score(sk_model.predict(X_test), y_test)

    # Train and evaluate sklearn logreg
    logreg = LogisticRegression(max_iter=max_iter)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        logreg.fit(X_train, y_train)
    lr_acc = accuracy_score(logreg.predict(X_test), y_test)

    n_classes = np.unique(y).shape[0]
    print(f"{n_classes}-class| Sklearn: {sk_acc:.3f}; Logreg: {lr_acc:.3f}")


if __name__ == "__main__":
    #two_class()
    ten_class()
