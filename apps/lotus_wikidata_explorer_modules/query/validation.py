"""
SMILES validation and escaping utilities.
"""

from typing import Tuple, Optional

__all__ = ["validate_smiles", "escape_smiles_for_sparql"]


def validate_smiles(smiles: str) -> Tuple[bool, Optional[str]]:
    """
    Validate SMILES string for common issues.

    Args:
        smiles: SMILES string to validate

    Returns:
        Tuple of (is_valid, error_message)
        If valid, error_message is None
    """
    # Empty is valid (means no SMILES search)
    if not smiles or not smiles.strip():
        return True, None

    smiles = smiles.strip()

    # Length validation
    if len(smiles) < 1:
        return False, "SMILES string is empty after trimming whitespace"
    if len(smiles) > 10000:
        return False, (
            f"SMILES string is too long ({len(smiles):,} characters). "
            f"Maximum allowed: 10,000 characters. "
            f"Please use a simpler structure or substructure."
        )

    # Check for null bytes or dangerous control characters
    if "\x00" in smiles:
        return False, "SMILES contains null bytes (\\x00) which are not allowed"

    invalid_chars = [c for c in smiles if ord(c) < 32 and c not in "\t\n\r"]
    if invalid_chars:
        chars_display = ", ".join(f"\\x{ord(c):02x}" for c in invalid_chars[:3])
        return False, (
            f"SMILES contains invalid control characters: {chars_display}. "
            f"Only standard ASCII printable characters are allowed."
        )

    # Basic sanity check: should contain at least one atom symbol
    # Common atom symbols in SMILES: C, N, O, S, P, F, Cl, Br, I, B, c (aromatic), etc.
    if not any(c in smiles for c in "CNOSPFIBcnops"):
        return False, (
            "SMILES appears to be missing atom symbols. "
            "A valid SMILES must contain at least one element (C, N, O, etc.). "
            "Example: 'c1ccccc1' for benzene"
        )

    return True, None


def escape_smiles_for_sparql(smiles: str) -> str:
    """
    Escape SMILES string for safe use in SPARQL queries.

    SMILES strings can contain backslashes (e.g., /C=C\\3/) which are escape
    characters in SPARQL string literals and must be escaped.

    Args:
        smiles: SMILES string to escape

    Returns:
        Escaped SMILES string safe for SPARQL

    Raises:
        ValueError: If SMILES is invalid
    """
    if not smiles:
        return smiles

    # Validate SMILES
    is_valid, error_msg = validate_smiles(smiles)
    if not is_valid:
        raise ValueError(f"Invalid SMILES: {error_msg}")

    # Escape backslashes by doubling them
    return smiles.replace("\\", "\\\\")
