import numpy as np

class Perceptron():
    def __init__(self, learning_rate=1e-1, max_iter=200):
        """
        A perceptron classifier. This binary classifier learns a linear
        boundary that separates input space into two, such that points
        on one side of the line are one class and points on the other side are
        the other class.

        Args:
            max_iter (int): the perceptron learning algorithm stops after
            this many iterations if it has not converged.

            learning_rate (float): how large of a step to take at each update

        """
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """
        Fit the perceptron to the data. You should not have to modify this
        function -- all your work should go in `update_weights` and `predict`.

        Note: self.map_01_to_pm1 is called to use labels in {-1, 1}
        Note: self.add_intercept is called to add an intercept to the features

        Args:
            X (np.ndarray): a NxK array containing N examples each with K features.
            y (np.ndarray): a Nx1 array containing binary targets.
        Returns:
            n_iters: the number of iterations the model took to converge,
                or self.max_iter
        """
        X = self.add_intercept(X)
        y = self.map_01_to_pm1(y)
        self.weights = np.zeros(X.shape[1])

        for n_iters in range(1, 1 + self.max_iter):
            stop = self.update_weights(X, y)
            if stop:
                break

        return n_iters

    def map_01_to_pm1(self, y):
        """
        Helper function to map {0, 1} labels to {-1, 1} labels
        This is called in fit before updating weights
        """
        return np.where(y == 0, -1, y)

    def map_pm1_to_01(self, y):
        """
        Helper function to map {-1, 1} labels to {0, 1} labels
        As the model was trained to predict {-1, 1} labels, this should
            be called after self.weights is used to compute predictions
            but before those predictions are returned
        """
        return np.where(y == -1, 0, y)

    def add_intercept(self, X):
        """
        Helper function to add a column of 1's to your features
        """
        return np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)

    def update_weights(self, X, y):
        """
        Perform one iteration of updates for the Perceptron algorithm
        Note: don't forget to use `self.learning_rate`

        Pseudocode:
            for each example in X
                if the model misclassifies that example
                    update the weights to better classify that example
            return whether perceptron has converged

        Args:
            X: the Nx(K+1) matrix of features, including an intercept
            y: the Nx1 array of targets, converted to {-1, 1}

        Returns:
            Boolean indicating whether the Perceptron has converged
        """
        converged = True
        for i in range(X.shape[0]):
            prediction = np.sign(np.dot(X[i], self.weights))
            if prediction != y[i]:
                converged = False
                self.weights += self.learning_rate * y[i] * X[i]
        return converged

    def predict(self, X):
        """
        Given features, a 2D numpy array, use the trained model to predict
        target classes. Call this after calling fit.

        Note: Keep the `self.add_intercept` to ensure you include the intercept
        Note: you will have to use `self.map_pm1_to_01` to convert your
            predictions to {0, 1} so they match the labels.

        Args:
            X (np.ndarray): 2D array containing real-valued inputs.
        Returns:
            predictions (np.ndarray): Output of trained model on features,
                with predictions converted to {0, 1} labels.
        """

        X = self.add_intercept(X)
        predictions = np.sign(np.dot(X, self.weights))
        return self.map_pm1_to_01(predictions)
