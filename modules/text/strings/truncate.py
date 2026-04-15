"""String truncation."""

__all__ = ["truncate"]


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max_length, adding suffix if truncated.

    Parameters
    ----------
    text : str
        Text.
    max_length : int
        Max length.
    suffix : str
        Default is '...'.

    Returns
    -------
    str
        Return value produced by truncate.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix
