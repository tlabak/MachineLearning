import numpy as np
import warnings

from src.utils import softmax


class NaiveBayes:
    """
    A Naive Bayes classifier for binary data.
    """

    def __init__(self, smoothing=1):
        """
        Args:
            smoothing: controls the smoothing behavior when computing p(x|y).
                If the word "jackpot" appears `k` times across all documents with
                label y=1, we will instead record `k + self.smoothing`. Then
                `p("jackpot" | y=1) = (k + self.smoothing) / Z`, where Z is a
                normalization constant that accounts for adding smoothing to
                all words.
        """
        self.smoothing = smoothing

    def predict(self, X):
        """
        Return the most probable label for each row x of X.
        You should not need to edit this function.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        """
        Using self.alpha and self.beta, compute the probability p(y | X[i, :])
            for each row X[i, :] of X.  While you will have used log
            probabilities internally, the returned array should be
            probabilities, not log probabilities.

        See equation (9) in `naive_bayes.pdf` for a convenient way to compute
            this using your self.alpha and self.beta. However, note that
            (9) produces unnormalized log probabilities; you will need to use
            your src.utils.softmax function to transform those into probabilities
            that sum to 1 in each row.

        Args:
            X: a sparse matrix of shape `[n_documents, vocab_size]` on which to
               predict p(y | x)

        Returns 
            probs: an array of shape `[n_documents, n_labels]` where probs[i, j] contains
                the probability `p(y=j | X[i, :])`. Thus, for a given row of this array,
                np.sum(probs[i, :]) == 1.
        """
        n_docs, vocab_size = X.shape
        n_labels = 2

        assert hasattr(self, "alpha") and hasattr(self, "beta"), "Model not fit!"
        assert vocab_size == self.vocab_size, "Vocab size mismatch"

        #raise NotImplementedError
        log_prior = self.alpha
        log_likelihood = np.zeros((n_docs, n_labels))

        for k in range(n_labels):
            log_likelihood[:, k] = X @ self.beta[:, k]



        log_posterior = log_prior + log_likelihood
        #probs = softmax(log_posterior)
        probs = softmax(log_posterior, axis=1)
        return probs

    def fit(self, X, y):
        """
        Compute self.alpha and self.beta using the training data.
        You should store log probabilities to avoid underflow.
        This function *should not* use unlabeled data. Wherever y is NaN, that
        label and the corresponding row of X should be ignored.

        See equations (10) and (11) in `naive_bayes.pdf` for the math necessary
            to compute your alpha and beta.

        self.alpha should be set to contain the marginal probability of each class label.

        self.beta should be set to the conditional probability of each word
            given the class label: p(w_j | y_i). This should be an array of shape
            [n_vocab, n_labels].  Remember to use `self.smoothing` to smooth word counts!
            See __init__ for details. If we see M total words across all N documents with
            label y=1, have a vocabulary size of V words, and see the word "jackpot" `k`
            times, then: `p("jackpot" | y=1) = (k + self.smoothing) / (M + self.smoothing *
            V)` Note that `p("jackpot" | y=1) + p("jackpot" | y=0)` will not sum to 1;
            instead, `sum_j p(word_j | y=1)` will sum to 1.

        Hint: when self.smoothing = 0, some elements of your beta will be -inf.
            If `X_{i, j} = 0` and `\beta_{j, y_i} = -inf`, your code should
            compute `X_{i, j} \beta_{j, y_i} = 0` even though numpy will by
            default compute `0 * -inf` as `nan`.

            This behavior is important to pass both `test_smoothing` and
            `test_tiny_dataset_a` simultaneously.

            The easy way to do this is to leave `X` as a *sparse array*, which
            will solve the problem for you. You can also explicitly define the
            desired behavior, or use `np.nonzero(X)` to only consider nonzero
            elements of X.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: None; sets self.alpha and self.beta
        """
        n_docs, vocab_size = X.shape
        n_labels = 2
        self.vocab_size = vocab_size

        #raise NotImplementedError

        # Ignore unlabeled rows
        mask = ~np.isnan(y)
        X, y = X[mask], y[mask]

        # Compute alpha
        self.alpha = np.zeros(n_labels)
        for k in range(n_labels):
            self.alpha[k] = np.log(np.sum(y == k) / n_docs) # Eq.(10) in naive_bayes.pdf

        # Compute beta
        self.beta = np.zeros((vocab_size, n_labels))
        for k in range(n_labels):
            rows = (y == k)
            count = X[rows].sum(axis=0) + self.smoothing
            self.beta[:, k] = np.log(count / count.sum()) # Eq.(11) in naive_bayes.pdf

        # Convert X to array
        #X = X.toarray()

        # Store the sparse matrix
        self.X = X
                

    def likelihood(self, X, y):
        """
        Using fit self.alpha and self.beta, compute the log likelihood of the data.
            You should use logs to avoid underflow.
            This function should not use unlabeled data. Wherever y is NaN,
            that label and the corresponding row of X should be ignored.

        Equation (5) in `naive_bayes.pdf` contains the likelihood, which can be written:

            \sum_{i=1}^N \alpha_{y_i} + \sum_{i=1}^N \sum_{j=1}^V X_{i, j} \beta_{j, y_i}

            You can visualize this formula in http://latex2png.com

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: the log likelihood of the data
        """
        assert hasattr(self, "alpha") and hasattr(self, "beta"), "Model not fit!"

        n_docs, vocab_size = X.shape
        n_labels = 2


        log_alpha = self.alpha
        log_beta = self.beta

        # Compute the log likelihood of the data
        log_likelihood = 0
        for i in range(n_docs):
            if not np.isnan(y[i]):
                # Add the log probability of the class label
                log_likelihood += log_alpha[y[i].astype(int)]

                # Add the log probability of the words given the class label
                for j in range(vocab_size):
                    if X[i, j] > 0:
                        log_likelihood += X[i, j] * log_beta[j, y[i].astype(int)]
                    else:
                        log_likelihood += 0

        # Return the log likelihood
        return log_likelihood