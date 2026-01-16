"""Escape SMILES for SPARQL."""

__all__ = ["escape_for_sparql"]


def escape_for_sparql(smiles: str) -> str:
    """Escape SMILES string for safe use in SPARQL queries."""
    if not smiles:
        return smiles
    return smiles.replace("\\", "\\\\")
