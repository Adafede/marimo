"""Normalize QID to uppercase."""

__all__ = ["normalize"]


def normalize(value: str) -> str:
    """Normalize a QID to uppercase.

    Parameters
    ----------
    value : str
        Value.

    Returns
    -------
    str
        String representation of normalize.

    """
    return value.strip().upper()
