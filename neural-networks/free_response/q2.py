import datetime
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src import load_data, plt
from src import Model, FullyConnected, Regularizer
from src import SigmoidActivation, ReluActivation
from src import BinaryCrossEntropyLoss
from src import custom_transform

from free_response.visualize import plot_decision_regions


def visualize_spiral():
    """
    Helper function to help visualize the spiral dataset
    """
    X, y, _ = load_data("data/spiral.csv")
    axis = plt.subplot()
    axis.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
    plt.close("all")


def visualize_transform():
    """
    Helper function to help visualize your data transformation
    """

    X, y, _ = load_data("data/spiral.csv")
    new_X = custom_transform(X)
    fig = plt.figure()

    # If 2D, plot 2D
    if new_X.shape[1] == 2:
        axis = plt.axes()
        axis.scatter(new_X[:, 0], new_X[:, 1], c=y)
    else:
        # Else plot 3D
        assert new_X.shape[1] == 3
        axis = fig.add_subplot(projection='3d')
        y1 = y == 1
        axis.scatter(new_X[y1, 0], new_X[y1, 1], new_X[y1, 2], c='r', depthshade=False)
        y0 = y == 0
        axis.scatter(new_X[y0, 0], new_X[y0, 1], new_X[y0, 2], marker='*', c='b', depthshade=False)

    plt.show()
    plt.close("all")


def main():
    """
    Load the spiral dataset and classify it with a MLP.
    This will show a plot and save it to your `free_reponse/` folder.
    With the initial hyperparameters below, this takes ~15s to run on Zach's laptop.
    """

    # Try changing the layer structure and hyperparameters
    n_iters = 10000
    learning_rate = 0.5
    reg = Regularizer(alpha=0, penalty="l2")
    layers = [
        FullyConnected(2, 254, regularizer=reg),
        SigmoidActivation(),
        FullyConnected(254, 1, regularizer=reg),
        SigmoidActivation()
    ]

    # Don't change the following code
    X, y, _ = load_data("data/spiral.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = Model(layers, BinaryCrossEntropyLoss(), learning_rate)
    model.fit(X_train, y_train, n_iters)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6),
                             sharex=True, sharey=True)
    acc = 100 * accuracy_score(model.predict(X_train) > 0.5, y_train)
    plot_decision_regions(
        X_train, y_train, model, title=f"Train accuracy {acc:.1f}%", axis=axes[0])

    acc = 100 * accuracy_score(model.predict(X_test) > 0.5, y_test)
    plot_decision_regions(
        X_test, y_test, model, title=f"Test accuracy {acc:.1f}%", axis=axes[1])
    fn = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.png")
    plt.savefig(f"free_response/q2_{fn}")
    plt.show()


if __name__ == "__main__":
    #visualize_spiral()
    #visualize_transform()
    main()
