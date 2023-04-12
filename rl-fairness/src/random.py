import itertools
import numpy as np
import os
import src


class RandomNumberGenerator:
    """
    Differences in random number generation can make test cases hard to debug
    This class loads pre-generated random numbers to hopefully make it easier
    """
    def __init__(self):
        self.seed()

    def seed(self):
        """
        Reload the random numbers from file for reproducibility 
        """
        self.uniform_iter = self.get_uniform_iter()
        self.normal_iter = self.get_normal_iter()

    def get_uniform_iter(self):
        """
        Load from file the uniformly-distributed floats
        """
        dirname = os.path.dirname
        root = dirname(dirname(os.path.abspath(src.__file__)))
        inf = os.path.join(root, "uniforms.npy")
        uniforms = np.load(inf, allow_pickle=False)
        return iter(uniforms)

    def next_uniform(self):
        """
        Return the next float generated from a uniform distribution
        """
        try:
            f = next(self.uniform_iter)
        except StopIteration:
            self.uniform_iter = self.get_uniform_iter()
            f = next(self.uniform_iter)
        finally:
            return f

    def get_normal_iter(self):
        """
        Load from file the normally-distributed floats
        """
        dirname = os.path.dirname
        root = dirname(dirname(os.path.abspath(src.__file__)))
        inf = os.path.join(root, "normals.npy")
        normals = np.load(inf, allow_pickle=False)
        return iter(normals)

    def next_normal(self):
        """
        Return the next float generated from a normal distribution
        """
        try:
            f = next(self.normal_iter)
        except StopIteration:
            self.normal_iter = self.get_normal_iter()
            f = next(self.normal_iter)
        finally:
            return f


def _one_randint(low, high, rng):
    """
    Internal call for getting a random integer
    """
    return int(low + rng.next_uniform() * (high - low))


def _one_rand(low, high, rng):
    """
    Internal call for getting a random float
    """
    return low + rng.next_uniform() * (high - low)


def _one_normal(loc, scale, rng):
    """
    Internal call for getting a single normal sample
    """
    return loc + rng.next_normal() * scale


def randint(low, high=None, size=None):
    """
    Replacement for numpy's `randint()`. Attempts to implement the same
        behavior, with the relevant numpy's docs copied below.

    Return random integers from low (inclusive) to high (exclusive).

    Return random integers from the “discrete uniform” distribution of the
    specified dtype in the “half-open” interval [low, high). If high is None (the
    default), then results are from [0, low).

    Parameters
    ----------
    low : int
        Lowest integers to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is one above the
        *highest* such integer).
    high : int, optional
        If provided, one above the largest integer to be drawn
        from the distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.

    Returns
    -------
    out : int or ndarray of ints
        `size`-shaped array of random integers from the appropriate
        distribution, or a single such random int if `size` not provided.
    """

    if high is None:
        high = low
        low = 0

    if size is None:
        return _one_randint(low, high, rng)

    elif type(size) == int:
        vals = [_one_randint(low, high, rng) for _ in range(size)]
        return np.array(vals, dtype=int)

    else:
        retval = np.zeros(size, dtype=int)
        for idxs in itertools.product(*[range(dim) for dim in size]):
            retval[idxs] = _one_randint(low, high, rng)
        return retval


def uniform(low, high=None, size=None):
    """
    Replacement for numpy's `uniform()`. Attempts to implement all the same
        behavior, with numpy's docs copied below.

    Draw samples from a uniform distribution.

    Samples are uniformly distributed over the half-open interval
    ``[low, high)`` (includes low, but excludes high).  In other words,
    any value within the given interval is equally likely to be drawn
    by `uniform`.

    Parameters
    ----------
    low : float or array_like of floats, optional
        Lower boundary of the output interval.  All values generated will be
        greater than or equal to low.  The default value is 0.

    high : float or array_like of floats
        Upper boundary of the output interval.  All values generated will be
        less than high.  The high limit may be included in the returned array of
        floats due to floating-point rounding in the equation
        ``low + (high-low) * random_sample()``.  high - low must be
        non-negative.  The default value is 1.0.

    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``low`` and ``high`` are both scalars.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized uniform distribution.
    """

    if high is None:
        high = low
        low = 0

    assert high > low, f"uniform() requires high > low, but {low} >= {high}"

    if size is None:
        return _one_rand(low, high, rng)

    elif type(size) == int:
        return np.array([_one_rand(low, high, rng) for _ in range(size)])

    else:
        retval = np.zeros(size)
        for idxs in itertools.product(*[range(dim) for dim in size]):
            retval[idxs] = _one_rand(low, high, rng)
        return retval


def choice(arr, size=None, replace=True, p=None):
    """
    Replacement for numpy's `choice()`. Implements the basic numpy
        functionality, but not the keyword arguments. The relevant
        numpy documentation is copied below.

    Generates a random sample from a given 1-D array

    arr : 1-D array-like or int
            If an ndarray, a random sample is generated from its elements.
            If an int, the random sample is generated as if it were ``np.arange(arr)``

    Returns
        samples : single item sampled from arr
    """
    if size is not None or p is not None or not replace:
        # NOTE: Do not implement anything in this function!
        # This just indicates that these keyword args which are defined for
        # numpy's choice() but are not implemented in this codebase.
        raise NotImplementedError(
            "Sorry, src.random.choice doesn't support that keyword argument")

    if type(arr) == int:
        length = arr
        arr = np.arange(arr)
    elif type(arr) in [list, tuple]:
        assert len(arr) > 0, "Can't call src.random.choice() on empty list/tuple"
        length = len(arr)
    elif type(arr) == np.ndarray:
        assert np.product(arr.shape) > 0, "Can't call src.random.choice() on empty array"
        assert len(arr.shape) == 1, "src.random.choice expects a 1-D array"
        length = arr.shape[0]
    else:
        raise NotImplementedError(
            f"Can't call src.random.choice() on arr of type {type(arr)}")

    idx = _one_randint(0, length, rng)
    return arr[idx]


def rand(size=None):
    """
    Replacement for numpy's `rand()`. Attempts to implement all the same
        behavior, which is just a simple wrapper for uniform.
    """
    return uniform(low=0, high=1, size=size)


def normal(loc=0, scale=1, size=None):
    """
    Replacement for numpy's `normal`. Attempts to implement all the same
        behavior. Relevant numpy documentation is copied below.  

    Draw random samples from a normal (Gaussian) distribution.

    Parameters
    ----------
    loc : float
        Mean ("centre") of the distribution.
    scale : float 
        Standard deviation (spread or "width") of the distribution. Must be
        non-negative.
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized normal distribution.
    """
    if size is None:
        return _one_normal(loc, scale, rng)

    elif type(size) == int:
        return np.array([_one_normal(loc, scale, rng) for _ in range(size)])

    else:
        retval = np.zeros(size)
        for idxs in itertools.product(*[range(dim) for dim in size]):
            retval[idxs] = _one_normal(loc, scale, rng)
        return retval


rng = RandomNumberGenerator()
