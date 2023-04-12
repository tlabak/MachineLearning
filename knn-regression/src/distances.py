import numpy as np


def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    X_norm = np.sum(np.square(X), axis=1, keepdims=True)
    Y_norm = np.sum(np.square(Y), axis=1, keepdims=True)
    dot = np.dot(X, Y.T)
    D = np.sqrt(X_norm + Y_norm.T - 2 * dot + 1e-10)
    return D


def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    return np.sum(np.abs(X[:, np.newaxis, :] - Y[np.newaxis, :, :]), axis=-1)


def cosine_distances(X, Y):
    """Compute pairwise Cosine distance between the rows of two matrices X (shape MxK)
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.

    (Hint: You're free to implement this with numpy.linalg.norm)

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """
    dot = np.dot(X, Y.T)
    X_norm = np.sqrt(np.sum(np.square(X), axis=1))
    Y_norm = np.sqrt(np.sum(np.square(Y), axis=1))
    return 1 - dot / (X_norm[:, np.newaxis] * Y_norm[np.newaxis, :] + 1e-10)
