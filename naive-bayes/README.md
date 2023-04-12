# CS349 HW 4: Naive Bayes and the EM Algorithm
Naive Bayes is a probabilistic machine learning algorithm used for classification tasks. The algorithm is based on Bayes' theorem, which states that the probability of a hypothesis (in this case, a label or class) is proportional to the product of the prior probability of the hypothesis and the conditional probability of the data given the hypothesis.

Naive Bayes is "naive" in the sense that it makes the assumption that the features are independent of each other given the class, which simplifies the calculations and makes the algorithm computationally efficient. Naive Bayes is widely used in text classification tasks, such as spam filtering and sentiment analysis.

The EM (Expectation-Maximization) algorithm is a method for estimating parameters of statistical models, particularly in cases where there are missing or incomplete data. The EM algorithm iteratively computes two steps: the E-step, which computes the expected value of the complete data log-likelihood given the observed data and the current estimates of the parameters, and the M-step, which maximizes the expected value of the complete data log-likelihood with respect to the parameters.

The EM algorithm is used in a variety of machine learning applications, such as clustering, mixture modeling, and latent variable models. It is particularly useful when dealing with incomplete or missing data, as it provides a way to estimate the missing values and learn the parameters of the model simultaneously.

## Coding
- Solved some simple practice problems with sparse matrices
- Wrote a stable softmax and log sum functions
- Implemented a fully-supervised NaiveBayes classifier
- Implemented a semi-supervised NaiveBayes classifier
