import pytest
import re
import warnings

import numpy as np

import src.numpy_practice
from tests.docstrings import docstrings


def check_function(function, max_length, required=[], prohibited=[]):
    """
    This function is designed to force you to get to know numpy,
        by preventing you from passing the test by writing for loops or if statements,
        and by restricting you to only `max_length` number of lines.
    """
    docstring = docstrings[function].split("\n")

    with open(src.numpy_practice.__file__) as infile:
        for line in infile:
            if "import" in line:
                assert line.strip() == "import numpy as np", "Don't import anything, sorry"

    with open(src.numpy_practice.__file__) as infile:
        for line in infile:
            if line.strip().startswith("def "):
                func = line.strip().split("def ")[1].split("(")[0]
                assert func in docstrings.keys(), f"You can solve these without writing {func}()"

    with open(src.numpy_practice.__file__) as infile:
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


def test_hello_world():
    retval = src.numpy_practice.hello_world()
    msg = "hello_world() should return \"Hello, world!\""
    assert retval == "Hello, world!", msg
    

@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
def test_numpy_replace_nonfinite_in_place():

    pairs = [
        (np.array([1, 2, 3, np.inf, 4, 5, np.nan, np.true_divide(-1, 0), 6]),
         np.array([1, 2, 3, 0, 4, 5, 0, 0, 6])),
        
        (np.array([-1, 1, 0, -1, 1, 0]) / np.array([1, 0, 1, 0, 1, 0]),
         np.array([-1, 0, 0, 0, 1, 0])),
    ]

    for before, after in pairs:
        retval = src.numpy_practice.replace_nonfinite_in_place(before)
        assert retval is None, "replace_nonfinite_in_place should return None"
        assert np.array_equal(before, after), f"{before} should equal {after}"

    check_function(
        "replace_nonfinite_in_place",
        1,
        required=[("np.isfinite", ), ("np.logical_not", "np.invert", "~")],
        prohibited=["for ", "if ", ";", "eval"]
    )


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
def test_numpy_replace_nans_out_of_place():

    pairs = [
        (np.array([1, 2, 3, np.inf, 4, 5, np.nan, np.true_divide(-1, 0), 6]),
         np.array([1, 2, 3, np.inf, 4, 5, 0, -np.inf, 6])),
        
        (np.array([-1, 1, 0, -1, 1, 0]) / np.array([1, 0, 1, 0, 1, 0]),
         np.array([-1, np.inf, 0, -np.inf, 1, 0])),
    ]

    for before, after in pairs:
        before_copy = before.copy()
        retval = src.numpy_practice.replace_nans_out_of_place(before)
        assert np.array_equal(retval, after), "replace_nans_out_of_place should return answer"
        assert np.array_equal(before, before_copy, equal_nan=True), f"{before} shouldn't be modified"

    check_function(
        "replace_nans_out_of_place",
        1,
        required=[("np.isnan", ), ("np.where")],
        prohibited=["for ", "if ", ";", "eval"]
    )


def test_numpy_find_mode():
    pairs = [
        (np.array([1, 2, 2, 3, 3, 3]),
         3),
        (np.concatenate([np.arange(3, 7), np.arange(6, 0, -1), np.arange(1, 7, 3)]),
         4),
        (np.concatenate([np.zeros(10), np.ones(11)]),
         1)
    ]
    for arr, target in pairs:
        arr = arr.astype(int)
        assert src.numpy_practice.find_mode(arr) == target, f"Mode is {target}" 

    check_function(
        "find_mode",
        3,
        required=[("np.argmax", ), ("np.unique", "np.bincount")],
        prohibited=["for ", "if ", ";", "eval"]
    )


def test_numpy_flip_and_slice_matrix():

    pairs = [
        (np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]),
         np.array([[2, 1, 0], [11, 10, 9]])),
        (np.array([[ 2,  6, 16, 18, 13, 13, 17]]).T,
         np.array([[2, 18, 17]]).T),
        (np.array([[ 0, 17, 14,  4]]),
         np.array([[ 4, 14, 17,  0]]))
    ]
    for before, after in pairs:
        msg = f"flip_and_slice_matrix({before}) should return {after}"
        before = before.astype(int)
        assert np.array_equal(src.numpy_practice.flip_and_slice_matrix(before), after), msg

    check_function(
        "flip_and_slice_matrix",
        2,
        required=[],
        prohibited=["for ", "if ", ";", "eval"]
    )


def test_numpy_divide_matrix_along_rows():

    trios = [
        (np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]),
         np.array([1, 2, 4]),
         np.array([[0, 1, 2, 3], [2, 2.5, 3, 3.5], [2, 2.25, 2.5, 2.75]])),
        (np.array([[2, 14], [2, 17], [8, 8], [12, 9], [9, 6], [10, 4]]),
         np.array([1, 1, 2, 3, 3, 2]),
         np.array([[2, 14], [2, 17], [4, 4], [4, 3], [3, 2], [5, 2]]))
    ]
    for (x, y, target) in trios:
        msg = f"divide_matrix_along_rows({x}, {y}) should return {target}"
        assert np.array_equal(src.numpy_practice.divide_matrix_along_rows(x, y), target), msg

    check_function(
        "divide_matrix_along_rows",
        2,
        required=[("reshape", "np.newaxis")],
        prohibited=["for ", "if ", ";", "eval"]
    )
