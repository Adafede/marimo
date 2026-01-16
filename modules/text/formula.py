"""
Molecular formula parsing and matching - no external dependencies.

Pure functions for parsing and analyzing molecular formula strings.
"""

__all__ = [
    "parse_formula",
    "count_element",
    "normalize_subscripts",
    "matches_element_range",
    "matches_halogen_constraint",
]

import re
from typing import Dict, Optional, Tuple

# Subscript to digit translation table
_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

# Compiled regex for element parsing
_ELEMENT_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")


def normalize_subscripts(formula: str) -> str:
    """
    Convert subscript digits to regular digits.

    Args:
        formula: Formula string possibly containing subscripts

    Returns:
        Formula with subscripts converted to regular digits

    Example:
        >>> normalize_subscripts("C₆H₁₂O₆")
        'C6H12O6'
    """
    return formula.translate(_SUBSCRIPT_MAP)


def parse_formula(formula: str) -> Dict[str, int]:
    """
    Parse molecular formula into element counts.

    Args:
        formula: Molecular formula string (e.g., "C6H12O6" or "C₆H₁₂O₆")

    Returns:
        Dictionary mapping element symbols to counts

    Example:
        >>> parse_formula("C6H12O6")
        {'C': 6, 'H': 12, 'O': 6}
        >>> parse_formula("NaCl")
        {'Na': 1, 'Cl': 1}
    """
    if not formula:
        return {}

    normalized = normalize_subscripts(formula)
    matches = _ELEMENT_PATTERN.findall(normalized)

    return {
        element: int(count) if count else 1 for element, count in matches if element
    }


def count_element(formula: str, element: str) -> int:
    """
    Count occurrences of an element in a formula.

    Args:
        formula: Molecular formula string
        element: Element symbol (e.g., "C", "Na")

    Returns:
        Count of the element, or 0 if not present

    Example:
        >>> count_element("C6H12O6", "C")
        6
        >>> count_element("C6H12O6", "N")
        0
    """
    atoms = parse_formula(formula)
    return atoms.get(element, 0)


def matches_element_range(
    formula: str,
    element: str,
    min_count: Optional[int] = None,
    max_count: Optional[int] = None,
) -> bool:
    """
    Check if element count in formula is within specified range.

    Args:
        formula: Molecular formula string
        element: Element symbol to check
        min_count: Minimum count (inclusive), None for no minimum
        max_count: Maximum count (inclusive), None for no maximum

    Returns:
        True if count is within range

    Example:
        >>> matches_element_range("C6H12O6", "C", min_count=5, max_count=10)
        True
        >>> matches_element_range("C6H12O6", "C", min_count=10)
        False
    """
    count = count_element(formula, element)

    if min_count is not None and count < min_count:
        return False
    if max_count is not None and count > max_count:
        return False
    return True


def matches_halogen_constraint(
    formula: str,
    halogen: str,
    constraint: str,
) -> bool:
    """
    Check if halogen presence matches constraint.

    Args:
        formula: Molecular formula string
        halogen: Halogen symbol (F, Cl, Br, I)
        constraint: One of "allowed", "required", "excluded"

    Returns:
        True if formula matches the constraint

    Example:
        >>> matches_halogen_constraint("C6H5Cl", "Cl", "required")
        True
        >>> matches_halogen_constraint("C6H6", "Cl", "required")
        False
        >>> matches_halogen_constraint("C6H6", "Cl", "excluded")
        True
    """
    count = count_element(formula, halogen)

    if constraint == "required":
        return count > 0
    elif constraint == "excluded":
        return count == 0
    else:  # "allowed"
        return True
