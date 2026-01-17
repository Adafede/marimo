"""Validate QID format."""

__all__ = ["is_valid"]


def is_valid(value: str) -> bool:
    """Check if a string is a valid Wikidata QID format (Q followed by digits)."""
    if not value:
        return False
    value = value.strip().upper()
    return value.startswith("Q") and value[1:].isdigit()
