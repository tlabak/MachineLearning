import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from src import PolynomialRegression, KNearestNeighbor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def visualize_model(data, model, title):
    """
    This function produces a plot with the training and test datasets,
    as well as the predictions of the trained model. The plot is saved
    to `free_response/` as a png.

    Note: You should not need to change this function!

    Args:
        data (tuple of np.ndarray): four arrays containing, in order:
            training data, test data, training targets, test targets
        model: the model with a .predict() method
        title: the title for the figure

    Returns:
        train_mse (float): mean squared error on training data
        test_mse (float): mean squared error on test data
    """

    X_train, X_test, y_train, y_test = data
    model.fit(X_train, y_train)
    x_func = np.arange(-1.2, 1.2, 0.01).reshape(-1, 1)
    preds = model.predict(x_func)

    train_mse = mean_squared_error(
        y_train, model.predict(X_train))
    test_mse = mean_squared_error(
        y_test, model.predict(X_test))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharey=True)

    fig.suptitle(title)
    ax1.set_title(f"Train MSE: {train_mse:.2f}",
                  fontdict={"fontsize": "medium"})
    ax1.set_ylim(-20, 20)
    ax1.scatter(X_train, y_train)
    ax1.plot(x_func, preds, "orange", label="h(X)")
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.legend()

    ax2.set_title(f"Test MSE: {test_mse:.2f}",
                  fontdict={"fontsize": "medium"})
    ax2.set_ylim(-20, 20)
    ax2.scatter(X_test, y_test)
    ax2.plot(x_func, preds, "orange", label="h(X)")
    ax2.set_xlabel('X')
    ax2.legend()
    plt.savefig(f"free_response/{title}.png")

    return train_mse, test_mse


def part_a_plot():
    """
    This uses matplotlib to create an example plot that you can modify
    for your answers to FRQ1 part a.
    """

    # Create a plot with four subplots
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 8), sharex=True)

    fig.suptitle("Demo plot for Q1 part a")

    # One subplot for each amount of data
    amounts = [8, 16, 64, 256]

    # Each plot has four points on the X-axis, either for `degree` or `k`
    x_axis = [1, 2, 4, 16]

    for idx, amount in enumerate(amounts):
        axes[idx].set_title(
            f"{amount} data points", fontdict={"fontsize": "small"}, pad=3)

        # This is some made-up data for the demo
        # You should replace this with your experimental results.
        #train_mse = np.random.uniform(1, 10, len(x_axis))
        #test_mse = np.random.uniform(1, 10, len(x_axis))
        # Collect the train and test MSE from the experiments
        train_mse_knn = []
        test_mse_knn = []
        for amount in amounts:
            for neighbor in x_axis:
                knn_reg = KNearestNeighbor(neighbor, distance_measure='euclidean')
                knn_reg.fit(data[0][:amount], data[1][:amount])
                predict_train = knn_reg.predict(data[0][:amount])
                predict_test = knn_reg.predict(data[0][amount:])
                train_mse_knn.append(mean_squared_error(data[1][:amount], predict_train))
                test_mse_knn.append(mean_squared_error(data[1][amount:], predict_test))
        train_mse_poly = []
        test_mse_poly = []
        for amount in amounts:
            for degree in x_axis:
                poly_reg = PolynomialRegression(degree)
                poly_reg.fit(data[0][:amount], data[2][:amount])
                predict_train = poly_reg.predict(data[0][:amount])
                predict_test = poly_reg.predict(data[0][amount:])
                train_mse_poly.append(mean_squared_error(data[2][:amount], predict_train))
                test_mse_poly.append(mean_squared_error(data[2][amount:], predict_test))
        
        # Plot the train and test MSE for KNN
        for idx, amount in enumerate(amounts):
            start = idx * len(x_axis)
            end = start + len(x_axis)
            axes[idx].set_title(
                f"{amount} data points (KNN)", fontdict={"fontsize": "small"}, pad=3)


        # Plot a solid red line for train error
        axes[idx].plot(np.array(x_axis), train_mse, 'r-', label="Train")
        # Plot a dashed blue line for test error
        axes[idx].plot(np.array(x_axis), test_mse, 'b--', label="Test")

        axes[idx].set_ylabel('MSE')
        axes[idx].legend()

    axes[idx].set_xlabel('X axis')
    plt.savefig("free_response/demo_plot.png")


def load_frq_data(amount):
    '''
    Loads the data provided for this free-response question,
    with `amount` examples.

    Note: You should not need to change this function!

    Args:
        amount (int): the number of examples to include
        in the dataset. Should be one of 8, 16, 64, or 256

    Returns
        data (tuple of np.ndarray): four arrays containing, in order:
            training data, test data, training targets, test targets
    '''
    df = pd.read_csv(f"data/frq.{amount}.csv")
    x1 = df[["x"]].to_numpy()
    y1 = df[["y"]].to_numpy()
    
    return train_test_split(
        x1, y1, train_size=0.8, random_state=0, shuffle=False)


def polynomial_regression_experiment():
    """
    Run 16 experiments with fitting PolynomialRegression models
        of different degrees on datasets of different sizes.

    You will want to use the `load_frq_data` and `visualize_model`
        functions, and may want to add some print statements to
        collect data on the overall trends.
    """
    degrees = [1, 2, 4, 16]
    amounts = [8, 16, 64, 256]

    for amount in amounts:
        for degree in degrees:
            title = f"{degree}-degree Regression with {amount} points"
            data = load_frq_data(amount)
            reg = PolynomialRegression(degree)
            reg.fit(data[0], data[2])
            visualize_model(data, reg, title)
            predict_train = reg.predict(data[0])
            r2 = r2_score(data[2], predict_train)
            print(f"{title} has R^2 = {r2}")


def knn_regression_experiment():
    '''
    Run 16 experiments with fitting KNearestNeighbor models
        of different n_neighbors on datasets of different sizes.

    You will want to use the `load_frq_data` and `visualize_model`
        functions, and may want to add some print statements to
        collect data on the overall trends.

    Use Euclidean distance and Mean aggregation for all experiments.
    '''
    n_neighbors = [1, 2, 4, 8]
    amounts = [8, 16, 64, 256]
    for amount in amounts:
        for neighbor in n_neighbors:
            title = f"{neighbor}-NN with {amount} points"
            data = load_frq_data(amount)
            reg = KNearestNeighbor(neighbor, distance_measure='euclidean')
            #data[1]
            reg.fit(data[0], data[2])
            visualize_model(data, reg, title)
            predict_train = reg.predict(data[0])
            print(data[2].shape, predict_train.shape)
            predict_train = predict_train.reshape(-1, 1)
            r2 = r2_score(data[2], predict_train)
            print(f"{title} has R^2 = {r2}")


if __name__ == "__main__":
    polynomial_regression_experiment()
    knn_regression_experiment()
    part_a_plot()
    plt.close('all')
