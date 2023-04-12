import numpy as np

import src.random


class Model:
    """
    A wrapper class for a neural network composed of
    layers and a loss function
    """
    def __init__(self, layers, loss, learning_rate=1):
        """
        layers: a list of layers
            each must have a `forward` and `backward` function
        loss: the loss function to use when calling self.backward

        You should not need to edit this function.
        """
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate

    def predict(self, X):
        """
        Helper function to match the scikit-learn API

        You will not need to edit this function.
        """
        return self.forward(X)

    def forward(self, X):
        """
        Take the input and pass it forward through each layer of the network,
        using the `.forward()` function of each layer.

        Return the output of the final layer.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X
        
        

    def backward(self, pred, y):
        """
        Take the predicted and target outputs and compute the loss.

        Then, beginning with `self.loss` and continuing *backwards*
        through each layer of the network, use the `.backward()`
        function of each layer to perform backpropagation.

        Note: each call to `backward()` in self.layers
            should use self.learning_rate

        Returns None
        """
        """
        # Compute the gradient of the loss with respect to the predicted output
        loss_grad = self.loss.backward()
        
        # Backpropagate through each layer of the neural network
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, self.learning_rate)
        """
        # Compute the gradient of the loss with respect to the predicted output
        loss = self.loss.forward(pred, y)
        loss_grad = self.loss.backward()

        # Store the input values in self.input_
        self.loss.input_ = (pred, y)

        # Backpropagate through each layer of the neural network
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, self.learning_rate)


    def fit(self, X, y, max_iter=10000):
        """
        Train the model on the data for `max_iter` iterations.
        For each iteration, call `self.forward` and then `self.backward`
            to make a prediction and then update each layer's weights.

        This function should always run for `max_iter` iterations;
            don't stop even if the gradients are negligibly small.

        Returns None
        """
        """
        for i in range(max_iter):
            pred = self.forward(X)
            self.backward(pred, y)
        """
        loss = float('inf')  # initialize loss to infinity
        for i in range(max_iter):
            pred = self.forward(X)
            self.backward(pred, y)
            loss = self.loss.forward(pred, y)  # assign the loss inside the loop


def main():
    '''
    A simple MLP to fit the xor dataset.
    This should run and get 100% accuracy after you finish
        implementing the functions in this file.
    '''
    from src.layers import FullyConnected, SigmoidActivation
    from src.loss import BinaryCrossEntropyLoss
    src.random.rng.seed()

    # xor dataset
    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    layers = [
      FullyConnected(2, 8), SigmoidActivation(),
      FullyConnected(8, 1), SigmoidActivation(),
    ]

    model = Model(layers, BinaryCrossEntropyLoss(), learning_rate=0.1)
    model.fit(X, y, max_iter=10000)
    preds = model.forward(X)
    print("{:.0f}% accuracy".format(100 * np.mean((preds > 0.5) == y)))


if __name__ == "__main__":
    main()
