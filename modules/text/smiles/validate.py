"""Validate SMILES string."""

__all__ = ["validate"]

from typing import Tuple, Optional


def validate(
    smiles: str,
    max_length: int = 10000,
    required_atoms: str = "CNOSPFIBcnops",
) -> Tuple[bool, Optional[str]]:
    """Validate SMILES string for common issues."""
    if not smiles or not smiles.strip():
        return True, None

    smiles = smiles.strip()

    if len(smiles) < 1:
        return False, "SMILES string is empty after trimming whitespace"

    if len(smiles) > max_length:
        return False, f"SMILES string is too long ({len(smiles):,} characters)"

    if "\x00" in smiles:
        return False, "SMILES contains null bytes"

    invalid_chars = [c for c in smiles if ord(c) < 32 and c not in "\t\n\r"]
    if invalid_chars:
        return False, "SMILES contains invalid control characters"

    if not any(c in smiles for c in required_atoms):
        return False, "SMILES appears to be missing atom symbols"

    return True, None
