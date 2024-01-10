"""
A simple script to crossmatch arrays borrowed from Andrew Hearin's halotools.
"""
import numpy as np

def crossmatch(x, y, skip_bounds_checking=False):
    """
    Parameters
    ----------
    x : integer array
        Array of integers with possibly repeated entries.
    y : integer array
        Array of unique integers.
    skip_bounds_checking : bool, optional
        The first step in the `crossmatch` function is to test that the input
        arrays satisfy the assumptions of the algorithm
        (namely that ``x`` and ``y`` store integers,
        and that all values in ``y`` are unique).
        If ``skip_bounds_checking`` is set to True,
        this testing is bypassed and the function evaluates faster.
        Default is False.
    Returns
    -------
    idx_x : integer array
        Integer array used to apply a mask to x
        such that x[idx_x] == y[idx_y]
    y_idx : integer array
        Integer array used to apply a mask to y
        such that x[idx_x] == y[idx_y]
    """
    # Ensure inputs are Numpy arrays
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    # Require that the inputs meet the assumptions of the algorithm
    if skip_bounds_checking is True:
        pass
    else:
        try:
            assert len(set(y)) == len(y)
            assert np.all(np.array(y, dtype=np.int64) == y)
            assert np.shape(y) == (len(y), )
        except:
            msg = ("Input array y must be a 1d sequence of unique integers")
            raise ValueError(msg)
        try:
            assert np.all(np.array(x, dtype=np.int64) == x)
            assert np.shape(x) == (len(x), )
        except:
            msg = ("Input array x must be a 1d sequence of integers")
            raise ValueError(msg)

    # Internally, we will work with sorted arrays, and then undo the sorting at the end
    idx_x_sorted = np.argsort(x)
    idx_y_sorted = np.argsort(y)
    x_sorted = np.copy(x[idx_x_sorted])
    y_sorted = np.copy(y[idx_y_sorted])

    # x may have repeated entries, so find the unique values as well as their multiplicity
    unique_xvals, counts = np.unique(x_sorted, return_counts=True)

    # Determine which of the unique x values has a match in y
    unique_xval_has_match = np.in1d(unique_xvals, y_sorted, assume_unique=True)

    # Create a boolean array with True for each value in x with a match, otherwise False
    idx_x = np.repeat(unique_xval_has_match, counts)

    # For each unique value of x with a match in y, identify the index of the match
    matching_indices_in_y = np.searchsorted(y_sorted, unique_xvals[unique_xval_has_match])

    # Repeat each matching index according to the multiplicity in x
    idx_y = np.repeat(matching_indices_in_y, counts[unique_xval_has_match])

    # Undo the original sorting and return the result
    return idx_x_sorted[idx_x], idx_y_sorted[idx_y]
