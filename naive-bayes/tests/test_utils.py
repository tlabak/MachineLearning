import pytest
import numpy as np


def test_softmax():
    from src.utils import softmax

    X = np.array([[1, 2], [3, 4]])
    inp = softmax(X, axis=0)
    exp0 = np.array([[0.11920292, 0.11920292],
                     [0.88079708, 0.88079708]])
    assert np.allclose(inp, exp0), (inp, exp0)
    exp1 = np.array([[0.26894142, 0.73105858],
                     [0.26894142, 0.73105858]])
    inp = softmax(X, axis=1)
    assert np.allclose(inp, exp1), (inp, exp1)

    pairs = [
        ([[0.5, 0.5]],     [[0.5, 0.5]]),
        ([[-1e30, -1e30]], [[0.5, 0.5]]),
        ([[1e30, 1e30]],   [[0.5, 0.5]]),
        ([[1e30, 1e29]],   [[1.0, 0.0]]),
        ([[-1e30, -1e30]], [[0.5, 0.5]]),
        ([[-1e30, -1e29]], [[0.0, 1.0]]),
        ([[-12, -13]],     [[0.73105858, 0.26894142]]),
    ]

    for inp, expected in pairs:
        est_ = softmax(np.array(inp), axis=1)
        err = (inp, est_)
        # The softmax output should sum to 1
        assert np.isclose(np.sum(est_), 1), err
        # The softmax should match reference
        assert np.allclose(est_, np.array(expected)), err

        # Repeat these tests with axis=0 and transpose
        est_ = softmax(np.transpose(np.array(inp)), axis=0)
        assert np.isclose(np.sum(est_), 1), (inp, est_)
        assert np.allclose(est_, np.transpose(np.array(expected))), (inp, est_)

    # Specific behavior when you pass an array of shape [2, ];
    #     np.atleast_2d makes it shape [1, 2], so axis=1 is expected
    small = np.array([0, np.log(3)])
    inp = softmax(small, axis=1)
    exp = np.array([0.25, 0.75])
    assert np.allclose(inp, exp), (inp, exp)
    inp = softmax(small, axis=0)
    exp = np.array([1, 1])
    assert np.allclose(inp, exp)

    # An array
    X = np.array([
        [-3.10776032, -0.24210597,  1.54828807],
        [-1.34697582,  0.40519057,  1.25280003],
        [-0.41016295, -2.26109990,  0.12321099],
        [-1.98941508,  0.69304351,  3.60277884],
    ])

    # Array's expected softmax output along axis=0
    exp0 = np.array([
        [0.04045175, 0.1788668 , 0.10217091],
        [0.23530779, 0.34170163, 0.07603235],
        [0.60046674, 0.02375154, 0.02457107],
        [0.12377373, 0.45568003, 0.79722566],
    ])

    # Array's expected softmax output along axis=1
    exp1 = np.array([
        [0.00807885, 0.14186895, 0.85005220],
        [0.04943691, 0.28510673, 0.66545636],
        [0.34943482, 0.05489269, 0.59567249],
        [0.00352181, 0.05149242, 0.94498577],
    ])

    inp = softmax(X, axis=0)
    assert np.allclose(inp, exp0), (inp, exp0)
    inp = softmax(X, axis=1)
    assert np.allclose(inp, exp1), (inp, exp1)

    # Another array
    X = np.array([
        [-9300, -9592],
        [-9190, -9097],
        [-7970, -7759]])

    # Array's expected softmax output along axis=0
    exp0 = np.array([
        [0, 0],
        [0, 0],
        [1, 1]])

    # Array's expected softmax output along axis=1
    exp1 = np.array([
        [1, 0],
        [0, 1],
        [0, 1]])

    inp = softmax(X, axis=0)
    assert np.all(np.isfinite(inp)), "Should be finite"
    assert np.allclose(inp, exp0), (inp, exp0)

    inp = softmax(X, axis=1)
    assert np.all(np.isfinite(inp)), "Should be finite"
    assert np.allclose(inp, exp1), (inp, exp1)


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
def test_stable_log_sum():

    from src.utils import stable_log_sum

    X = np.array([
        [-2, -4],
        [-8, -2],
        [-2, -2],
        [-8, -4],
    ])

    pairs = [
        (16,    -160),
        (64,    -640),
        (256,   -2560),
        (1024,  -10240),
        (16384, -163840),
    ]

    for scale, target in pairs:
        Y = scale * X
        naive_calculation = np.sum(np.log(np.sum(np.exp(Y), axis=1)))
        retval = stable_log_sum(Y)

        if np.isfinite(naive_calculation):
            assert np.isclose(retval, naive_calculation, atol=1)

        assert np.isclose(retval, target, atol=1)
