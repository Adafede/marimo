"""
RDKit-based chemistry utilities.

Functions requiring RDKit for molecule processing and visualization.
"""

__all__ = [
    "find_mcs_smarts",
    "render_molecule_with_highlights",
]

from typing import Tuple, Optional
from collections import defaultdict


def find_mcs_smarts(
    smiles_list: list[tuple[str, str]],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find Maximum Common Substructure SMARTS from a list of SMILES.

    Args:
        smiles_list: List of (name, smiles) tuples

    Returns:
        Tuple of (smarts_string, error_message).
        If successful, error_message is None.
        If failed, smarts_string is None.

    Example:
        >>> find_mcs_smarts([("ethanol", "CCO"), ("methanol", "CO")])
        ('CO', None)
    """
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.rdFMCS import FindMCS

    mols = [MolFromSmiles(smi) for _, smi in smiles_list]
    mols = [mol for mol in mols if mol is not None]
    if len(mols) < 2:
        return None, "⚠️ Need at least two valid SMILES to find MCS."

    mcs_result = FindMCS(mols)
    if mcs_result.canceled or not mcs_result.smartsString:
        return None, "⚠️ Could not determine MCS."

    return mcs_result.smartsString, None


def render_molecule_with_highlights(
    name: str,
    smi: str,
    smarts_mols: list[tuple[str, str, object, tuple[float, float, float]]],
    match_counter: Optional[defaultdict] = None,
    width: int = 200,
    height: int = 200,
) -> str:
    """
    Render a molecule as SVG with highlighted substructures.

    Args:
        name: Display name for the molecule
        smi: SMILES string
        smarts_mols: List of (name, smarts, smarts_mol, rgb_color) tuples
        match_counter: Optional defaultdict to count matches per SMARTS
        width: SVG width in pixels
        height: SVG height in pixels

    Returns:
        HTML string with SVG and labels

    Example:
        >>> from rdkit.Chem import MolFromSmarts
        >>> smarts_mols = [("hydroxyl", "[OH]", MolFromSmarts("[OH]"), (0.5, 0.5, 1.0))]
        >>> html = render_molecule_with_highlights("ethanol", "CCO", smarts_mols)
    """
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DSVG
    from rdkit.Chem.rdDepictor import Compute2DCoords

    mol = MolFromSmiles(smi)
    if not mol:
        return f"<div style='color:red;'>🚫 Invalid SMILES: <code>{smi}</code></div>"

    Compute2DCoords(mol)
    atom_ids, colors, tooltips = [], {}, []

    for s_name, smarts, smarts_mol, color in smarts_mols:
        matches = mol.GetSubstructMatches(smarts_mol)
        if matches:
            for match in matches:
                atom_ids.extend(match)
                for idx in match:
                    colors[idx] = color
            tooltips.append(f"✅ {s_name}: {len(matches)} match(es)")
            if match_counter is not None:
                match_counter[s_name] += 1

    drawer = MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol, highlightAtoms=atom_ids, highlightAtomColors=colors)
    drawer.FinishDrawing()

    label = (
        f"<strong>{name}</strong><br><code>{smi}</code>"
        if name != smi
        else f"<code>{smi}</code>"
    )

    return (
        "<div style='display:inline-block; margin:12px; text-align:center; "
        "border:1px solid #eee; padding:10px; border-radius:8px; "
        "box-shadow: 2px 2px 5px rgba(0,0,0,0.1);'>"
        f"{drawer.GetDrawingText()}<br>{label}<br>"
        f"<small>{'<br>'.join(tooltips)}</small></div>"
    )
