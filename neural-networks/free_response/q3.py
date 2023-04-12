import argparse
import datetime
import numpy as np

from src import load_data
from src import Model, FullyConnected, Regularizer
from src import SigmoidActivation, ReluActivation
from src import BinaryCrossEntropyLoss
from src import plt
import src.random

from free_response.visualize import plot_decision_regions


def main():
    """
    Explore the loss landscape for a single parameter in a linear model

    run this as `python -m free_response.q3 <batch_size>`
    where `batch_size` is an integer greater than 0 and less than 11.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("batch_size", type=int)
    args = parser.parse_args()

    # Load polynomial classification data
    X, y, _ = load_data("data/polynomial.csv")
    N = X.shape[0]
    msg = f"Batch size must be > 0 and <= {N // 2}"
    assert args.batch_size > 0 and args.batch_size <= (N // 2), msg

    # Set up a logistic regression using our Model function
    learning_rate = 0.1
    layer = FullyConnected(2, 1)
    layer.weights = np.array([[0.5], [0.5], [0.0]])
    layers = [layer, SigmoidActivation()]
    model = Model(layers, BinaryCrossEntropyLoss(), learning_rate)

    # Shuffle
    src.random.rng.seed()
    idx = np.argsort(src.random.rand(N))
    X, y = X[idx], y[idx]

    # Loop through the batches in the data
    best_vals = []
    n_batches = int(np.ceil(N / args.batch_size))
    for batch in range(n_batches):
        # Set up the batch
        start = batch * args.batch_size
        end = start + args.batch_size
        X_batch = X[start:end]
        y_batch = y[start:end]

        # For a range of possible values for W1[2, 0],
        #   plot the loss landscape of that weight
        weight_vals = np.linspace(-2, 4, 20)
        losses = []
        for val in weight_vals:
            layer.weights[2, 0] = val
            loss = model.loss.forward(model.predict(X_batch), y_batch)
            losses.append(loss)

        best_vals.append(weight_vals[np.argmin(losses)])

        # Make a scatter plot of the loss landscape
        plt.scatter(weight_vals, losses)
        plt.title("Loss landscape")
        plt.xlabel("W[2, 0] val")
        plt.ylabel("Loss")
        plt.savefig(f"free_response/q3_batch_{batch}_of_{n_batches}.png")
        # plt.show()  # uncomment if you want these to show during runtime
        plt.close("all")

    print("Plots saved to free_response/")

    # Report which parameter values gave the lowest loss
    std = np.std(best_vals)
    best_vals_str = ", ".join([f"{x:.2f}" for x in best_vals])
    print(f"Lowest loss occurred at: {best_vals_str}")
    print(f"With standard deviation of {std:.2f}")


if __name__ == "__main__":
    main()
