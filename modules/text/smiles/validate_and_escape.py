"""Validate and escape SMILES for SPARQL."""

__all__ = ["validate_and_escape"]

from .validate import validate
from ..strings.escape_backslashes import escape_backslashes


def validate_and_escape(smiles: str | None) -> str | None:
    """Validate and escape SMILES string for SPARQL. Raises ValueError if invalid."""
    if not smiles:
        return smiles

    is_valid, error_msg = validate(smiles)
    if not is_valid:
        raise ValueError(f"Invalid SMILES: {error_msg}")

    return escape_backslashes(smiles)
