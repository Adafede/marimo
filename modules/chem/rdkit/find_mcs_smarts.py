"""Find Maximum Common Substructure SMARTS from SMILES list."""

__all__ = ["find_mcs_smarts"]

from .smarts.find_mcs import find_mcs
from .smiles.parse_many import parse_many


def find_mcs_smarts(
    smiles_list: list[tuple[str, str]],
) -> tuple[str | None, str | None]:
    """
    Find Maximum Common Substructure SMARTS from a list of SMILES.

    Args:
        smiles_list: List of (name, smiles) tuples

    Returns:
        Tuple of (smarts_string, error_message)
    """
    valid_mols = parse_many(smiles_list=smiles_list)
    return find_mcs(mols=valid_mols)
