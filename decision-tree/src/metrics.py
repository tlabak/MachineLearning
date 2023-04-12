import numpy as np
# Note: do not import additional libraries to implement these functions


def compute_confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the confusion matrix. The confusion
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    You do not need to implement confusion matrices for labels with more
    classes. You can assume this will always be a 2x2 matrix.

    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    # Matrix for determining whether or not a measurement was accurate to the true value
    true_positives = np.sum((predictions == 1) & (actual == 1))
    false_negatives = np.sum((predictions == 0) & (actual == 1))
    false_positives = np.sum((predictions == 1) & (actual == 0))
    true_negatives = np.sum((predictions == 0) & (actual == 0))

    confusion_matrix = np.array([[true_negatives, false_positives], [false_negatives, true_positives]])

    return confusion_matrix


def compute_accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the accuracy:

    Hint: implement and use the compute_confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    confusion_matrix = compute_confusion_matrix(actual, predictions)
    accuracy = (confusion_matrix[0,0] + confusion_matrix[1,1]) / confusion_matrix.sum()
    return accuracy



def compute_precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    You MUST account for edge cases in which precision or recall are undefined
    by returning np.nan in place of the corresponding value.

    Hint: implement and use the compute_confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output a tuple containing:
        precision (float): precision
        recall (float): recall
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    confusion_matrix = compute_confusion_matrix(actual, predictions)
    true_positives = confusion_matrix[1, 1]
    false_positives = confusion_matrix[0, 1]
    false_negatives = confusion_matrix[1, 0]
    #false_positives = confusion_matrix[0, 0]

    if (true_positives + false_positives) == 0:
        precision = np.nan
    else:
        precision = true_positives / (true_positives + false_positives)

    if (true_positives + false_negatives) == 0:
        recall = np.nan
    else:
        recall = true_positives / (true_positives + false_negatives)

    return precision, recall


def compute_f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the F1-measure:

    https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Because the F1-measure is computed from the precision and recall scores, you
    MUST handle undefined (NaN) precision or recall by returning np.nan. You
    should also consider the case in which precision and recall are both zero.

    Hint: implement and use the compute_precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    precision, recall = compute_precision_and_recall(actual, predictions)
    if np.isnan(precision) or np.isnan(recall):
        return np.nan
    elif (precision == 0) and (recall == 0):
        return 0
    else:
        f1_measure = 2 * (precision * recall) / (precision + recall)
        return f1_measure
