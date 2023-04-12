import warnings
import numpy as np

from src.utils import softmax, stable_log_sum
from src.naive_bayes import NaiveBayes


class NaiveBayesEM(NaiveBayes):
    """
    A NaiveBayes classifier for binary data, that uses both unlabeled and
        labeled data in the Expectation-Maximization algorithm

    Note that the class definition above indicates that this class
        inherits from the NaiveBayes class. This means it has the same
        functions as the NaiveBayes class unless they are re-defined in this
        function. In particular you should be able to call `self.predict_proba`
        using your implementation from `src/naive_bayes.py`.
    """

    def __init__(self, max_iter=10, smoothing=1):
        """
        Args:
            max_iter: the maximum number of iterations in the EM algorithm,
                where each iteration contains both an E step and M step.
                You should check for convergence after each iterations,
                e.g. with `np.isclose(prev_likelihood, likelihood)`, but
                should terminate after `max_iter` iterations regardless of
                convergence.
            smoothing: controls the smoothing behavior when computing p(x|y).
                If the word "jackpot" appears `k` times across all documents with
                label y=1, we will instead record `k + self.smoothing`. Then
                `p("jackpot" | y=1) = (k + self.smoothing) / Z`, where Z is a
                normalization constant that accounts for adding smoothing to
                all words.
        """
        self.max_iter = max_iter
        self.smoothing = smoothing

    def initialize_params(self, vocab_size, n_labels):
        """
        Initialize self.alpha such that
            `log p(y_i = k) = -log(n_labels)`
            for all k
        and initialize self.beta such that
            `log p(w_j | y_i = k) = -log(vocab_size)`
            for all j, k.

        """

        #raise NotImplementedError
        self.alpha = np.full((n_labels,), -np.log(n_labels))
        self.beta = np.full((vocab_size, n_labels), -np.log(vocab_size))

    def fit(self, X, y):
        """
        Compute self.alpha and self.beta using the training data.
        You should store log probabilities to avoid underflow.
        This function *should* use unlabeled data within the EM algorithm.

        During the E-step, use the NaiveBayes superclass self.predict_proba to
            infer a distribution over the labels for the unlabeled examples.
            Note: you should *NOT* replace the true labels with your predicted
            labels. You can use a `np.where` statement to only update the
            labels where `np.isnan(y)` is True.

        During the M-step, update self.alpha and self.beta, similar to the
            `fit()` call from the NaiveBayes superclass. However, when counting
            words in an unlabeled example to compute p(x | y), instead of the
            binary label y you should use p(y | x).

        For help understanding the EM algorithm, refer to the lectures and
            the handout. In particular, Figure 2 shows the algorithm for
            semi-supervised Naive Bayes.

        self.alpha should contain the marginal probability of each class label.

        self.beta should contain the conditional probability of each word
            given the class label: p(x | y). This should be an array of shape
            [n_vocab, n_labels].  Remember to use `self.smoothing` to smooth word counts!
            See __init__ for details. If we see M total
            words across all documents with label y=1, have a vocabulary size
            of V words, and see the word "jackpot" `k` times, then:
            `p("jackpot" | y=1) = (k + self.smoothing) / (M + self.smoothing * V)`
            Note that `p("jackpot" | y=1) + p("jackpot" | y=0)` will not sum to 1;
            instead, `sum_j p(word_j | y=1)` will sum to 1.

        Note: if self.max_iter is 0, your function should call
            `self.initialize_params` and then break. In each
            iteration, you should complete both an E-step and
            an M-step.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: None
        """
        n_docs, vocab_size = X.shape
        n_labels = 2
        self.vocab_size = vocab_size

        self.initialize_params(vocab_size, n_labels)
        #raise NotImplementedError
        # If max_iter is 0, initialize_params and break
        if self.max_iter == 0:
            self.initialize_params(vocab_size, n_labels)
            return

        # Iterate until convergence or until max_iter is reached
        for i in range(self.max_iter):
            # E-step: compute p(y|x) for unlabeled examples
            if np.isnan(y).any():
                # There are unlabeled examples
                p_y_given_x = self.predict_proba(X)
            else:
                p_y_given_x = None

            # M-step: compute alpha and beta
            word_counts = np.zeros((vocab_size, n_labels))
            class_counts = np.zeros(n_labels)

            for j in range(n_docs):
                if np.isnan(y[j]):
                    # Unlabeled example
                    p_y_given_x_j = p_y_given_x[j]
                    # Compute word counts using p(y|x) instead of binary y
                    word_counts += (X[j, :].reshape(-1, 1) * p_y_given_x_j.reshape(1, -1)).reshape(vocab_size, n_labels)
                    # Update class counts using p(y|x)
                    class_counts += p_y_given_x_j
                else:
                    # Labeled example
                    word_counts[:, int(y[j])] += X[j, :]
                    class_counts[int(y[j])] += 1

            # Update alpha and beta
            self.alpha = np.log(class_counts / class_counts.sum())
            self.beta = np.log((word_counts + self.smoothing) / (word_counts.sum(axis=0) + self.smoothing * vocab_size))




    def likelihood(self, X, y):
        """
        Using fit self.alpha and self.beta, compute the likelihood of the data.
            This function *should* use unlabeled data.
            This likelihood is defined in equation (14) of `naive_bayes.pdf`.

        For unlabeled data, we predict `p(y_i = y' | X_i)` using the
            previously-learned p(x|y, beta) and p(y | alpha).
            For labeled data, we define `p(y_i = y' | X_i)` as
            1 if `y_i = y'` and 0 otherwise; this is because for labeled data,
            the probability that the ith example has label y_i is 1.

        Following equation (14) in the `naive_bayes.pdf` writeup, the log
            likelihood of the data after t iterations can be written as:

            \sum_{i=1}^N \log \sum_{y'=1}^2 \exp(
                \log p(y_i = y' | X_i, \alpha, \beta) + \alpha_{y'}
                + \sum_{j=1}^V X_{i, j} \beta_{j, y'})

            You can visualize this formula in http://latex2png.com

            The tricky aspect of this likelihood is that we are simultaneously
            computing $p(y_i = y' | X_i, \alpha^t, \beta^t)$ to predict a
            distribution over our latent variables (the unobserved $y_i$) while
            at the same time computing the probability of seeing such $y_i$
            using $p(y_i =y' | \alpha^t)$.

            Note: In implementing this equation, it will help to use your
                implementation of `stable_log_sum` to avoid underflow. See the
                documentation of that function for more details.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: the log likelihood of the data.
        """

        assert hasattr(self, "alpha") and hasattr(self, "beta"), "Model not fit!"

        n_docs, vocab_size = X.shape
        n_labels = 2

        #raise NotImplementedError
        # Calculate the log probability of y using alpha
        log_p_y = self.alpha
        
        # Calculate the log probability of x given y using beta
        log_p_x_given_y = X @ self.beta
        
        # Calculate the joint probability of x and y
        log_joint = log_p_x_given_y + log_p_y
        
        # Calculate the log probability of x marginalizing over y
        log_p_x = stable_log_sum(log_joint)
        
        # Calculate the log likelihood of the data
        log_likelihood = np.sum(log_p_x)
        
        return log_likelihood
