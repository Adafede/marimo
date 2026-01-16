"""Normalize QID to uppercase."""

__all__ = ["normalize"]


def normalize(value: str) -> str:
    """Normalize a QID to uppercase."""
    return value.strip().upper()
