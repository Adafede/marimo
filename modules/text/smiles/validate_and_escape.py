"""Validate and format structure text as a SPARQL string literal."""

__all__ = ["validate_and_escape"]

from .validate import validate
from ..strings.escape_backslashes import escape_backslashes


def _looks_like_molfile(text: str) -> bool:
    """Heuristic check for Molfile V2000/V3000 blocks.

    Parameters
    ----------
    text : str
        Text.

    Returns
    -------
    bool
        Result looks like molfile.
    """
    upper = text.upper()
    return "M  END" in upper and (
        "V3000" in upper
        or "V2000" in upper
        or "M  V30 BEGIN CTAB" in upper
        or "BEGIN CTAB" in upper
    )


def validate_and_escape(smiles: str | None) -> str | None:
    """Validate structure text and return a SPARQL-safe string literal.

            Accepts one-line SMILES and multiline Molfile blocks (e.g., V3000).

    Parameters
    ----------
    smiles : str | None
        Smiles.

    Returns
    -------
    str | None
        Result validate and escape.
    """
    if not smiles:
        return smiles

    normalized = smiles.replace("\r\n", "\n").replace("\r", "\n")
    is_molfile = _looks_like_molfile(normalized)

    # Keep Molfile formatting exactly as pasted (leading spaces and blank lines matter).
    candidate = normalized if is_molfile else normalized.strip()
    is_valid, error_msg = validate(candidate)
    if not is_valid:
        raise ValueError(f"Invalid structure input: {error_msg}")

    escaped = escape_backslashes(candidate)

    if is_molfile or "\n" in candidate:
        # Match SACHEM examples that work with multiline query text.
        return f"'''{escaped}'''"

    escaped = escaped.replace('"', '\\"')

    return f'"{escaped}"'
