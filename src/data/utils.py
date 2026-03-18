"""Data utilities helpers.

This module contains small helpers used by data loaders and wrappers.
"""


def create_slices_cdl(first_slice: int, last_slice: int, sep: int = 1):
    """Create a list of slice indices from first to last (inclusive).

    Parameters
    ----------
    first_slice : int
        Index of the first slice (inclusive).
    last_slice : int
        Index of the last slice (inclusive).
    sep : int, optional
        Step between slices. Default is 1.

    Returns
    -------
    list[int]
        List of slice indices.
    """
    slices = list(range(first_slice, last_slice + 1, sep))
    return slices
