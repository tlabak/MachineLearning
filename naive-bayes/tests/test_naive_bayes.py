import pytest
import numpy as np
import warnings

from tests.utils import build_small_dataset

train_data, train_labels, test_data, test_labels = build_small_dataset()


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in subtract:RuntimeWarning")
def test_tiny_dataset_a():
    from src.naive_bayes import NaiveBayes
    help_test_tiny_dataset(NaiveBayes)


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in subtract:RuntimeWarning")
def test_tiny_dataset_b():
    from src.naive_bayes_em import NaiveBayesEM
    help_test_tiny_dataset(NaiveBayesEM)

def help_test_tiny_dataset(model):
    from src.utils import softmax
    from scipy.sparse import csr_matrix

    X = csr_matrix(np.array([
        [1, 2, 1, 0, 0],
        [0, 1, 0, 1, 1],
        [3, 0, 1, 0, 0],
    ]))
    y = np.array([0, 1, 0])
    nb = model(smoothing=0)
    nb.fit(X, y)

    # alpha should be of shape [n_labels]
    assert nb.alpha.shape == (2, )
    # beta should be of shape [n_vocab, n_labels]
    assert nb.beta.shape == (5, 2)

    inp = softmax(nb.alpha)
    exp = np.array([2/3, 1/3])
    assert np.allclose(inp, exp), f"{inp} != {exp}"
    inp = nb.alpha
    exp = np.log(exp)
    assert np.allclose(inp, exp), f"{inp} != {exp}"

    inp = softmax(nb.beta, axis=0)
    beta_target = np.transpose(np.array([
        [0.5, 0.25, 0.25, 0, 0],
        [0, 1/3, 0, 1/3, 1/3]]))
    assert np.allclose(inp, beta_target), f"{inp} != {beta_target}"

    msg = f"{nb.beta} != {np.log(beta_target)}"
    assert np.allclose(nb.beta, np.log(beta_target)), msg

    # Log likelihood should match reference output
    assert np.isclose(nb.likelihood(X, y), -13.523145537608109)

    nb = model(smoothing=1)
    nb.fit(X, y)

    # Predicted probabilities should match reference
    probs = np.array([[0.90636996, 0.09363004],
                      [0.14877954, 0.85122046],
                      [0.99078844, 0.00921156]])
    est = nb.predict_proba(X)
    assert np.all(np.isclose(est, probs)), f"{est} != {probs}"


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in subtract:RuntimeWarning")
def test_alpha_beta_normalized():
    from src.naive_bayes import NaiveBayes
    from src.naive_bayes_em import NaiveBayesEM
    
    for n in range(2, 10):
        X = np.ones([n, 2])
        y = np.ones([n])
        y[0] = 0

        for smoothing in (0, 1):
            nb = NaiveBayes(smoothing=smoothing)
            nb.fit(X, y)
            # Alpha should sum to 1
            est = np.sum(np.exp(nb.alpha))
            assert np.isclose(est, 1), f"{est} != 1"

            # Beta should sum to 1
            est = np.sum(np.exp(nb.beta), axis=0)
            assert np.allclose(est, np.ones_like(est)), f"{est} != 1"

            for max_iter in range(1, 4):
                nbem = NaiveBayesEM(smoothing=smoothing, max_iter=max_iter)
                nbem.fit(X, y)
                # Alpha should sum to 1
                est = np.sum(np.exp(nbem.alpha))
                assert np.isclose(est, 1), f"{est} != 1"

                # Beta should sum to 1
                est = np.sum(np.exp(nbem.beta), axis=0)
                assert np.allclose(est, np.ones_like(est)), f"{est} != 1"


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_smoothing():
    from src.naive_bayes import NaiveBayes
    from scipy.sparse import csr_matrix

    X = csr_matrix(np.array([
        [1, 2, 1, 0, 0],
        [0, 1, 0, 1, 1],
        [3, 0, 1, 0, 0],
    ]))
    train_y = np.array([0, 1, 0])
    nb = NaiveBayes(smoothing=0)
    nb.fit(X, train_y)
    test_y = np.array([1, 0, 1])

    # The log likelihood should be log(0) = -np.inf
    #   this happens because smoothing = 0, train_y != test_y,
    #   and some words only show up in one of the two documents.
    msg = "likelihood with smoothing=0 should be -inf"
    assert nb.likelihood(X, test_y) == -np.inf, msg

    prev_prob = -np.inf
    # smoothing_vals = [1, 2, 4, 1e100]
    # smoothed_beta_values = [-np.log(9)]
    # for i in range(len(smoothing_vals)):
    for smoothing in [1, 2, 4, 1e100]:
        nb = NaiveBayes(smoothing=smoothing)
        nb.fit(X, train_y)
        prob = np.mean(nb.predict_proba(X)[(0, 1), (1, 0)])

        # The probability of seeing the opposite class should keep
        #     increasing as we increase the smoothing parameter
        msg = f"With smoothing={smoothing}, expect {prob} > {prev_prob}"
        assert prob > prev_prob, msg
        prev_prob = prob

        msg = "likelihood with smoothing should be finite"
        assert np.isfinite(nb.likelihood(X, test_y)), msg

        # Zero beta values should be smoothed to log(1 / (8 + 5 * smoothing))
        target_val = np.log(smoothing) - np.log(8 + 5 * smoothing)
        assert np.isclose(nb.beta[4, 0], target_val), (nb.beta[4, 0], target_val)

    # When smoothing is near-infinite, probabilities should all be 0.5
    assert np.isclose(prob, 0.5)


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_without_em():
    from src.naive_bayes import NaiveBayes

    # Train and evaluate NB without EM
    nb = NaiveBayes()
    nb.fit(train_data, train_labels)
    nb_likelihood = nb.likelihood(train_data, train_labels)

    is_labeled = np.isfinite(train_labels)
    nb_preds = nb.predict(train_data[is_labeled, :])
    train_accuracy = np.mean(nb_preds == train_labels[is_labeled])

    # NB should get 100% accuracy on the two labeled examples
    assert train_accuracy == 1.0

    nb_probs = nb.predict_proba(train_data)
    # Predict_proba should output a [n_documents, n_labels] array
    assert nb_probs.shape == (train_labels.shape[0], 2)
    # Probabilities should sum to 1
    assert np.all(np.isclose(np.sum(nb_probs, axis=1), np.ones_like(train_labels)))


def test_em_initialization():
    from src.utils import softmax
    from src.naive_bayes_em import NaiveBayesEM

    nbem = NaiveBayesEM(max_iter=0)
    nbem.initialize_params(train_data.shape[1], 2)
    assert np.isclose(np.sum(softmax(nbem.alpha, axis=1)), 1)
    assert np.allclose(np.sum(softmax(nbem.beta, axis=0), axis=0), 1)

    nbem.fit(train_data, train_labels)

    # If you do zero EM steps, your initialized probabilities should be uniform
    assert np.all(nbem.alpha[0] == nbem.alpha)
    assert np.all(nbem.beta[0, :] == nbem.beta)


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_em_basics():
    from src.naive_bayes_em import NaiveBayesEM
    prev_likelihood = -np.inf

    train_data, train_labels, _, _ = build_small_dataset()
    train_labels2 = train_labels.copy()

    alphas = []
    betas = []
    max_iters = [1, 2, 3, 4, 5]
    for max_iter in max_iters:
        nbem = NaiveBayesEM(max_iter=max_iter)
        nbem.fit(train_data, train_labels)
        likelihood = nbem.likelihood(train_data, train_labels)

        # EM should only ever increase the likelihood
        assert likelihood >= prev_likelihood, "EM should increase likelihood"
        prev_likelihood = likelihood

        msg = "Don't overwrite y!"
        assert np.array_equal(train_labels, train_labels2, equal_nan=True), msg

        alphas.append(nbem.alpha.copy())
        betas.append(nbem.beta.copy())

    msg = "Each iteration should update alpha/beta"
    assert np.unique(alphas, axis=0).shape[0] == len(max_iters), msg
    assert np.unique(betas, axis=0).shape[0] == len(max_iters), msg


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_comparison_naive_bayes():
    from src.naive_bayes import NaiveBayes
    from src.naive_bayes_em import NaiveBayesEM

    # Train and evaluate NB without EM
    nb1 = NaiveBayes()
    nb1.fit(train_data, train_labels)
    nb1_likelihood = nb1.likelihood(train_data, train_labels)
    nb1_preds = nb1.predict(test_data)
    nb1_accuracy = np.mean(nb1_preds == test_labels)

    # Train and evaluate NB with EM
    nb2 = NaiveBayesEM()
    nb2.fit(train_data, train_labels)
    nb2_likelihood = nb2.likelihood(train_data, train_labels)
    nb2_preds = nb2.predict(test_data)
    nb2_accuracy = np.mean(nb2_preds == test_labels)

    # NB using EM should outperform NB without it
    assert nb2_accuracy > nb1_accuracy

    # NB with EM should have a lower likelihood. Why? We'll see in the FRQ
    assert nb2_likelihood < nb1_likelihood
