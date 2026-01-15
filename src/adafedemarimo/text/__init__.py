"""
Text utilities subpackage - no external dependencies.

Pure functions for string manipulation, SMILES processing, and formula parsing.
"""

from adafedemarimo.text.strings import (
    pluralize,
    truncate,
)

from adafedemarimo.text.smiles import (
    validate_smiles,
    escape_for_sparql,
)

from adafedemarimo.text.formula import (
    parse_formula,
    count_element,
    normalize_subscripts,
    matches_element_range,
    matches_halogen_constraint,
)

__all__ = [
    # strings
    "pluralize",
    "truncate",
    # smiles
    "validate_smiles",
    "escape_for_sparql",
    # formula
    "parse_formula",
    "count_element",
    "normalize_subscripts",
    "matches_element_range",
    "matches_halogen_constraint",
]

