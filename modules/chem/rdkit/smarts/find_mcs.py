"""Find Maximum Common Substructure SMARTS from molecules."""

__all__ = ["find_mcs"]

from rdkit.Chem import Mol
from rdkit.Chem.rdFMCS import FindMCS

MIN_MOLECULES: int = 2
ERROR_INSUFFICIENT: str = (
    f"[!] Need at least {MIN_MOLECULES} valid molecules to find MCS."
)
ERROR_FAILED: str = "[!] Could not determine MCS."


def find_mcs(mols: list[Mol]) -> tuple[str | None, str | None]:
    """
    Find Maximum Common Substructure SMARTS from a list of molecules.

    Args:
        mols: List of RDKit Mol objects

    Returns:
        Tuple of (smarts_string, error_message)
    """
    if len(mols) < MIN_MOLECULES:
        return None, ERROR_INSUFFICIENT

    mcs_result = FindMCS(mols)
    if mcs_result.canceled or not mcs_result.smartsString:
        return None, ERROR_FAILED

    return mcs_result.smartsString, None
