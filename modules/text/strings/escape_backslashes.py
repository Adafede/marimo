"""Escape backslashes."""

__all__ = ["escape_backslashes"]


def escape_backslashes(str: str) -> str:
    """Escape backslashes from string."""
    return str.replace("\\", "\\\\")
