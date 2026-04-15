"""Parse multiple SMILES strings to RDKit Mol objects."""

__all__ = ["parse_many"]

from rdkit.Chem import Mol, MolFromSmiles


def parse_many(smiles_list: list[tuple[str, str]]) -> list[Mol]:
    """Parse list of (name, smiles) tuples to valid RDKit Mol objects.

            Filters out invalid SMILES that cannot be parsed.
            Uses generator for memory efficiency.

    Parameters
    ----------
    smiles_list : list[tuple[str, str]]
        Smiles list.

    Returns
    -------
    list[Mol]
        Return value produced by parse many.
    """
    return [mol for _, smi in smiles_list if (mol := MolFromSmiles(smi)) is not None]
