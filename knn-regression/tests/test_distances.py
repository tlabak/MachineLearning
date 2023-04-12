import src
import numpy as np
import src.random

def test_euclidean_distances():
    from sklearn.metrics.pairwise import euclidean_distances
    src.random.rng.seed()
    x = src.random.rand(size=(97, 37))
    y = src.random.rand(size=(83, 37))
    _est = src.euclidean_distances(x, y)
    assert _est.shape == (97, 83), "Distance matrix shape is wrong"
    _true = euclidean_distances(x, y)
    assert (np.allclose(_true, _est)), "Euclidean calculations off"


def test_manhattan_distances():
    from sklearn.metrics.pairwise import manhattan_distances
    src.random.rng.seed()
    x = src.random.rand(size=(97, 37))
    y = src.random.rand(size=(83, 37))
    _est = src.manhattan_distances(x, y)
    assert _est.shape == (97, 83), "Distance matrix shape is wrong"
    _true = manhattan_distances(x, y)
    assert (np.allclose(_true, _est)), "Manhattan calculations off"


def test_cosine_distances():
    from sklearn.metrics.pairwise import cosine_distances
    src.random.rng.seed()
    x = src.random.uniform(-1, 1, size=[97, 37])
    y = src.random.uniform(-1, 1, size=[83, 37])
    _est = src.cosine_distances(x, y)
    assert _est.shape == (97, 83), "Distance matrix shape is wrong"
    assert np.all(_est >= 0), "Cosine *distance* should be non-negative"
    _true = cosine_distances(x, y)
    assert (np.allclose(_true, _est)), "Cosine calculations off"

    for multiplier in [10, 100, 1000]:
        y_ = y * multiplier
        _est = src.cosine_distances(x, y_)
        assert (np.allclose(_true, _est)), "Cosine calculations off"
