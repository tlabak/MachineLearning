import numpy as np
from src.numpy_practice import find_mode


class PredictMode():
    def __init__(self):
        """
        This is a simple classifier that just looks at the labels in the dataset
        and learns to always predict the mode (most common) label.

        For example:
            >>> features = np.ones([6, 1])
            >>> model = PredictMode()
            >>> model.fit(features, np.array([1, 2, 2, 3, 3, 3]))
            >>> model.predict(features)
            np.array([3, 3, 3, 3, 3, 3])

        """
        self.most_common_class = None

    def fit(self, features, labels):
        """
        Looking at the 
        Implement a classifier that works by prior probability. Takes in features
        and labels and fits the features to the labels using prior probability.
        Looking at the provided labels, record the mode (most common) label.

        You may call your `find_mode` function from `src.numpy_practice`

        Args:
            features (np.array): numpy array of shape (n, d)
                 where n is number of examples and d is number of features.
            labels (np.array): numpy array containing true labels for each of the N
                examples.
        Output:
            None: Simply update self.most_common_class with the most common label
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        mode = unique_labels[np.argmax(counts)]
        self.most_common_class = mode

    def predict(self, features):
        """
        Predicts classes for each example in `features` using the trained model.
        Note that for PredictMode, this function won't actually use the values of `features`.

        Args:
            features (np.array): numpy array of shape (n, d)
                 where n is number of examples and d is number of features.
        Outputs:
            predictions (np.array): numpy array of size (n, ) which contains
                one predicted label per feature
        """
        predictions = np.full(features.shape[0], self.most_common_class)
        return predictions
