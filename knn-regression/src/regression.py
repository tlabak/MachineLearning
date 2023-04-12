import numpy as np
import src.random


class PolynomialRegression():
    def __init__(self, degree):
        """
        Implement PolynomialRegression from scratch.
        
        The `degree` argument controls the complexity of the function.  For
        example, degree = 2 would specify a hypothesis space of all functions
        of the form:

            f(x) = ax^2 + bx + c

        You should implement the closed form solution of least squares:
            w = (X^T X)^{-1} X^T y
        
        Do not import or use these packages: scipy, sklearn, sys, importlib.
        Do not use these numpy or internal functions: polynomial, polyfit, polyval, getattr, globals

        Args:
            degree (int): Degree used to fit the data.
        """
        self.degree = degree
        self.weights = None
    
    def fit(self, features, targets):
        """
        Fit to the given data.

        Hints:
          - Remember to use `self.degree`
          - Remember to include an intercept (a column of all 1s) before you
            compute the least squares solution.
          - If you are getting `numpy.linalg.LinAlgError: Singular matrix`,
            you may want to compute a "pseudoinverse" or add a tiny bit of
            random noise to your input data.

        Args:
            features (np.ndarray): an array of shape [N, 1] containing real-valued inputs.
            targets (np.ndarray): an array of shape [N, 1] containing real-valued targets.
        Returns:
            None (saves model weights to `self.weights`)
        """
        # Add an intercept column of ones to features
        features = np.concatenate((np.ones((features.shape[0], 1)), features), axis=1)
        
        # Generate feature matrix with poly. features
        X = np.zeros((features.shape[0], self.degree + 1))
        for i in range(self.degree + 1):
            X[:, i:i + 1] = np.power(features[:, 1:2], i)
        
        # weights computations, closed form solution
        try:
            self.weights = np.linalg.inv(X.T @ X) @ X.T @ targets
        except np.linalg.LinAlgError:
            # Add a tiny bit of random noise to X to make it invertible
            X += src.random.normal(0, 1e-10, X.shape)
            self.weights = np.linalg.inv(X.T @ X) @ X.T @ targets

    def predict(self, features):
        """
        Given features, use the trained model to predict target estimates. Call
        this only after calling fit so that the model has its weights.

        Args:
            features (np.ndarray): array of shape [N, 1] containing real-valued inputs.
        Returns:
            predictions (np.ndarray): array of shape [N, 1] containing real-valued predictions
        """
        assert hasattr(self, "weights"), "Model hasn't been fit!"

        N = features.shape[0]
        X = np.ones((N, 1))
        for i in range(1, self.degree + 1):
            X = np.hstack((X, features**i))
        return X @ self.weights
