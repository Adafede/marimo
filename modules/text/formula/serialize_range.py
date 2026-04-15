"""Serialize ElementRange to dictionary."""

__all__ = ["serialize_range"]

from .element_range import ElementRange


def serialize_range(element_range: ElementRange) -> dict[str, int | None] | None:
    """Convert ElementRange to dictionary. Returns None if not active.

    Parameters
    ----------
    element_range : ElementRange
        Element range.

    Returns
    -------
    dict[str, int | None] | None
        Return value produced by serialize range.
    """
    if not element_range.is_active():
        return None
    return {"min": element_range.min_val, "max": element_range.max_val}
