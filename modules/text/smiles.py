"""
SMILES string validation and processing - no external dependencies.

Pure functions for working with SMILES chemical notation strings.
"""

__all__ = [
    "validate_smiles",
    "escape_for_sparql",
]

from typing import Tuple, Optional


def validate_smiles(
    smiles: str,
    max_length: int = 10000,
    required_atoms: str = "CNOSPFIBcnops",
) -> Tuple[bool, Optional[str]]:
    """
    Validate SMILES string for common issues.

    Args:
        smiles: The SMILES string to validate
        max_length: Maximum allowed length
        required_atoms: Characters that indicate valid atom symbols

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.

    Example:
        >>> validate_smiles("c1ccccc1")
        (True, None)
        >>> validate_smiles("")
        (True, None)  # Empty is valid (means no SMILES search)
        >>> validate_smiles("123")
        (False, 'SMILES appears to be missing atom symbols...')
    """
    if not smiles or not smiles.strip():
        return True, None

    smiles = smiles.strip()

    if len(smiles) < 1:
        return False, "SMILES string is empty after trimming whitespace"

    if len(smiles) > max_length:
        return False, (
            f"SMILES string is too long ({len(smiles):,} characters). "
            f"Maximum allowed: {max_length:,} characters."
        )

    if "\x00" in smiles:
        return False, "SMILES contains null bytes (\\x00) which are not allowed"

    invalid_chars = [c for c in smiles if ord(c) < 32 and c not in "\t\n\r"]
    if invalid_chars:
        chars_display = ", ".join(f"\\x{ord(c):02x}" for c in invalid_chars[:3])
        return False, (
            f"SMILES contains invalid control characters: {chars_display}. "
            f"Only standard ASCII printable characters are allowed."
        )

    if not any(c in smiles for c in required_atoms):
        return False, (
            "SMILES appears to be missing atom symbols. "
            "A valid SMILES must contain at least one element (C, N, O, etc.)."
        )

    return True, None


def escape_for_sparql(smiles: str) -> str:
    """
    Escape SMILES string for safe use in SPARQL queries.

    SMILES strings can contain backslashes (e.g., /C=C\\3/) which are
    escape characters in SPARQL string literals and must be doubled.

    Args:
        smiles: The SMILES string to escape

    Returns:
        The escaped SMILES string

    Example:
        >>> escape_for_sparql("C/C=C\\\\C")
        'C/C=C\\\\\\\\C'
    """
    if not smiles:
        return smiles
    return smiles.replace("\\", "\\\\")
