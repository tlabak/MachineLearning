docstrings = {
    "hello_world": None,
    "replace_nonfinite_in_place": '''""" 
    In the given array, replace (in-place!) all non-finite values with the value 0.
    You should be able to do this in a single line of code.
    You may not use a `for` loop or `if` statement!

    For example:
        >>> x = np.array([1, 2, 3, np.inf, 4, 5, np.nan, -1 / 0, 6])
        >>> replace_nans_in_place(x)
        >>> x
        array([1, 2, 3, 0, 4, 5, 0, 0, 6])

    You should use:
        - np.isfinite: https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
    You should use one of the following:
        - np.logical_not: https://numpy.org/doc/stable/reference/generated/numpy.logical_not.html
        - np.invert: https://numpy.org/doc/stable/reference/generated/numpy.invert.html
        - The tilde (~) syntax, which works as shorthand for np.invert.
    Args:
        x: a numpy array
    Returns:
        None: you can explicitly call `return None`, but Python does so automatically
              if you don't return anything else
    """''',

    "replace_nans_out_of_place":  '''"""
    In the given array, replace (not in-place!) all nans with the value 0.
    You should be able to do this in a single line of code.
    You may not use a `for` loop or `if` statement!

    For example:
        >>> x = np.array([1, 2, 3, np.inf, 4, 5, np.nan, -1 / 0, 6])
        >>> replace_nans_in_place(x)
        >>> x
        array([1, 2, 3, inf, 4, 5, 0, -inf, 6])
    
    You should use:
        - np.isnan: https://numpy.org/doc/stable/reference/generated/numpy.isnan.html
        - np.where: https://numpy.org/doc/stable/reference/generated/numpy.where.html

    Args:
        x: a numpy array
    Returns:
        a numpy array where all NaNs (but not infinite) values are replaced with 0s
    """''',

    "find_mode": '''"""
    In the given array, find the mode of the vector.
    The mode is the value that appears the most times (don't worry about ties).

    You should be able to do this in two or fewer lines of code.
    You may not use a `for` loop or `if` statement!

    For example:
        >>> x = np.array([1, 2, 2, 3, 3, 3])
        >>> find_mode(x)
        3

    You should use:
        - np.argmax: https://numpy.org/doc/stable/reference/generated/numpy.argmax.html

    You should use one of:
        - np.unique: https://numpy.org/doc/stable/reference/generated/numpy.unique.html
        - np.bincount: https://numpy.org/doc/stable/reference/generated/numpy.bincount.html

    Args:
        x: a numpy array of integers
    Returns:
        the mode of x
    """''',

    "flip_and_slice_matrix": '''"""
    Take the matrix x and flip it horizontally, then take the every third row.

    You should be able to do this in two or fewer lines of code.
    You may not use a `for` loop or `if` statement!

    For example:
        >>> x = np.array([[ 0,  1,  2],
        ...               [ 3,  4,  5],
        ...               [ 6,  7,  8],
        ...               [ 9, 10, 11],
        ...               [12, 13, 14]])
        >>> flip_and_slice_matrix(x)
        array([[2, 1, 0],
               [11, 10, 9]])

    First, read:
        - https://numpy.org/doc/stable/user/basics.indexing.html#basics-indexing
        
    Args:
      x: a matrix
    Returns:
      a numpy matrix
    """''',

    "divide_matrix_along_rows": '''"""
    Take the matrix x and divide it by the vector y, such that
        the ith row of x is divided by the ith value of y.

    You should be able to do this in two or fewer lines of code.
    You may not use a `for` loop or `if` statement!

    For example:
        >>> x = np.array([[ 0,  1,  2,  3],
        ...               [ 4,  5,  6,  7],
        ...               [ 8,  9, 10, 11]])
        >>> y = np.array([1, 2, 4])
        >>> divide_rows(x, y)
        array([[0.  , 1.  , 2.  , 3.  ],
               [2.  , 2.5 , 3.  , 3.5 ],
               [2.  , 2.25, 2.5 , 2.75]])

    First, read:
        - https://numpy.org/doc/stable/user/basics.broadcasting.html
    You should use one of:
        - np.reshape: https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
        - np.newaxis: https://numpy.org/doc/stable/reference/constants.html#numpy.newaxis

    Args:
      x: a matrix
      y: a vector with as many entries as x has rows
    Returns:
      a numpy matrix
    """'''
}
