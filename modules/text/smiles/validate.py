"""Validate SMILES string."""

__all__ = ["validate"]

DEFAULT_MAX_LENGTH: int = 10000
DEFAULT_REQUIRED_ATOMS: str = "CNOSPFIBcnops"


def validate(
    smiles: str,
    max_length: int = DEFAULT_MAX_LENGTH,
    required_atoms: str = DEFAULT_REQUIRED_ATOMS,
) -> tuple[bool, str | None]:
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
