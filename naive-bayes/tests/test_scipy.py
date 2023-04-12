import timeit
import pytest
import re
import warnings

import numpy as np
import scipy.sparse

import src.sparse_practice
from tests.docstrings import docstrings


def check_function(function, max_length, required=[], prohibited=[]):
    """
    This function is designed to force you to get to know numpy/scipy,
        by preventing you from passing the test by writing for loops or if statements,
        and by restricting you to only `max_length` number of lines.
    """
    docstring = docstrings[function].split("\n")

    exceptions = ["import numpy as np"]
    for line in docstring:
        if "import" in line:
            exceptions.append(line.strip())

    with open(src.sparse_practice.__file__, encoding="utf-8") as infile:
        for line in infile:
            if "import" in line:
                assert line.strip() in exceptions, "Don't import anything, sorry"

    with open(src.sparse_practice.__file__, encoding="utf-8") as infile:
        for line in infile:
            if line.strip().startswith("def "):
                func = line.strip().split("def ")[1].split("(")[0]
                assert func in docstrings.keys(), f"You can solve these without writing {func}()"

    with open(src.sparse_practice.__file__, encoding="utf-8") as infile:
        # Seek to the beginning of the function we're checking
        for line in infile:
            if line.strip().startswith(f"def {function}"):
                break

        # Make sure the docstring is unchanged
        for i, doc_line in enumerate(docstring):
            line = infile.readline()
            assert line.strip() == doc_line.strip(), f"Don't change the {i}th line of the docstring in function {function}()"

        # Get the lines actually used to write this function
        #   iterate to the next function or end of file
        lines = []
        for line in infile:
            if line.strip().startswith("def"):
                break
            line = line.strip()

            # don't count blank and commented-out lines towards the total
            if len(line) > 0 and not line.startswith("#"):
                lines.append(line)

        if max_length == 1:
            msg = "just one line"
        else:
            msg = f"in {max_length} lines or fewer"
        assert len(lines) <= max_length, f"You should write {function}() in {msg}"

        # Make sure prohibited functions weren't used
        d = {key: 0 for key in required}
        for line in lines:
            for key in prohibited:
                assert key not in line, f"You can't use `{key}` in {function}()"

            for key_group in d:
                for key in key_group:
                    if key in line:
                        d[key_group] += 1

        # Make sure required functions were used
        for key_group, val in d.items():
            if len(key_group) == 1:
                msg = f"Your code for {function}() must use {key_group[0]}"
            else:
                lst = ", ".join(key_group)
                msg = f"Your code for {function}() must use one of: {lst}"
            assert val > 0, msg


@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def test_scipy_sparse_to_numpy():

    # Some example numpy arrays
    arrays = [
        np.arange(20).reshape(5, 4),
        np.random.normal(0, 5, size=[10, 10]),
        np.random.uniform(-10, 10, size=[100, 100])
    ]

    # Convert numpy array to sparse matrix; your code should convert it back
    for arr in arrays:
        sparse = scipy.sparse.csr_matrix(arr)
        retval = src.sparse_practice.sparse_to_numpy(sparse)
        assert np.array_equal(arr, retval), f"{arr} should equal {retval}"

    check_function(
        "sparse_to_numpy",
        1,
        required=[],
        prohibited=["for ", "while", "if ", ";", "eval"]
    )


@pytest.mark.filterwarnings("ignore::scipy.sparse.SparseEfficiencyWarning")
def test_scipy_sparse_multiplication():

    # Zeros in a sparse matrix are not actually used in calculations,
    #   so 0 * np.inf gets computed as 0 instead of np.nan
    X = scipy.sparse.csr_matrix(np.array([
            [1, 1, 1, 0, 0],
            [0, 1, 0, 1, 1]]))
    Y = np.array([
            [1, 1, 1, np.inf, np.inf],
            [np.inf, 1, np.inf, 1, 1]])
    Z = np.array([[3, np.inf], [np.inf, 3]])

    out = src.sparse_practice.sparse_multiplication(X, Y.T)
    assert np.all(out == Z), f"{out.reshape(-1)} != {Z.reshape(-1)}"

    # Zeros in a sparse matrix are not actually used in calculations,
    #   so 0 * np.inf gets computed as 0 instead of np.nan
    size = 10
    X = scipy.sparse.csr_matrix(np.ones([size, 1]))
    Y = np.ones([size])
    for idx in range(size):
        X[idx] = 0
        X.eliminate_zeros()  # "re-introduce sparsity"
        Y[idx] = np.nan
        
        out = src.sparse_practice.sparse_multiplication(X.T, Y)
        assert np.sum(np.isnan(out)) == 0, out

    # Sparse multiplication should be extremely fast 
    for size in [1000, 10000, 100000]:
        X = scipy.sparse.csr_matrix((size, size), dtype=int)
        Y = scipy.sparse.csr_matrix((size, size), dtype=int)
        idx = np.random.randint(0, size)
        X[idx, idx] = 1
        Y[idx, idx] = 1

        func = src.sparse_practice.sparse_multiplication
        d = {'func': func, 'np': np, 'X': X, 'Y': Y}
        runtime = timeit.timeit("assert np.sum(func(X, Y)) == 1", number = 10, globals = d) 
        assert runtime < 1, f"Should be fast; took {runtime:.1f} seconds"

    check_function(
        "sparse_multiplication",
        1,
        required=[],
        prohibited=["for ", "while", "if ", ";", "eval"]
    )
