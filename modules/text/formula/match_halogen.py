"""Check halogen constraint matching."""

__all__ = ["match_halogen"]

from .count_element import count_element


def match_halogen(formula: str, halogen: str, constraint: str) -> bool:
    """Check if halogen presence matches constraint (allowed/required/excluded)."""
    cnt = count_element(formula, halogen)
    if constraint == "required":
        return cnt > 0
    elif constraint == "excluded":
        return cnt == 0
    return True  # "allowed"
