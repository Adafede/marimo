"""Escape backslashes."""

__all__ = ["escape_backslashes"]


def escape_backslashes(str: str) -> str:
    """Escape backslashes from string.

Parameters
----------
str : str
    Str.

Returns
-------
str
    Computed result.
    """
    return str.replace("\\", "\\\\")
