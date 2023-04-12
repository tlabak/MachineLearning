from src.decision_tree import DecisionTree
from src.predict_mode import PredictMode
from src.data import load_data, train_test_split

import numpy as np


def run(data_path, learner_type, fraction, **kwargs):
    """
    This function walks through an entire machine learning workflow as follows:

        1. takes in a path to a dataset
        2. loads it into a numpy array with `load_data`
        3. instantiates the class used for learning from the data using learner_type (e.g
           learner_type is 'decision_tree', 'predict_mode')
        4. splits the data into training and testing with `train_test_split` and `fraction`.
        5. trains a learner using the training split with `fit`
        6. tests the trained learner using the testing split with `predict`
        7. evaluates the trained learner's accuracy on the test set

    Each run of this function constitutes a trial. Your learner should be pretty
    robust across multiple runs, as long as `fraction` is sufficiently high. See how
    unstable your learner gets when less and less data is used for training by
    playing around with `fraction`.

    IMPORTANT:
    If fraction == 1.0, then your training and testing sets should be exactly the
    same. This is so that the test cases are deterministic. Test cases with fraction == 1.0
    check whether you fit the training data correctly, not whether you generalize to
    a testing set.

    Args:
        data_path (str): path to csv file containing the data
        learner_type (str): either 'decision_tree' or 'predict_mode'.
            For each of these, the associated learner is instantiated and used
            for the experiment.
        fraction (float between 0.0 and 1.0): fraction of examples to be drawn for training

    Returns:
        accuracy (np.float): Accuracy on testing examples using learner
    """

    # load data into numpy array
    X, Y, attributes = load_data(data_path)

    # instantiate class used for learning from the data
    if learner_type == 'decision_tree':
        learner = DecisionTree(attributes)
    elif learner_type == 'predict_mode':
        learner = PredictMode()
    else:
        raise ValueError(f'Invalid learner type {learner_type}')

    # split the data into training and testing
    X_train, Y_train, X_test, Y_test = train_test_split(X, Y, fraction)

    # train learner using the training split, predict on test split
    learner.fit(X_train, Y_train)
    Y_pred = learner.predict(X_test)

    # Return accuracy
    return np.mean(Y_test == Y_pred)
