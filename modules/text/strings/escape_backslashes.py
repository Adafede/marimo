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
        Return value produced by escape backslashes.
    """
    return str.replace("\\", "\\\\")
