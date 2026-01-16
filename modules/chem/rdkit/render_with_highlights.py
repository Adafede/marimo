"""Render molecule as SVG with highlighted substructures."""

__all__ = ["render_with_highlights"]

from collections import defaultdict
from typing import Any

# RDKit imports are inside the function for lazy loading (optional dependency)

_INVALID_SMILES_TEMPLATE = (
    "<div style='color:red;'>🚫 Invalid SMILES: <code>{smi}</code></div>"
)

_SVG_CONTAINER_STYLE = (
    "display:inline-block; margin:12px; text-align:center; "
    "border:1px solid #eee; padding:10px; border-radius:8px; "
    "box-shadow: 2px 2px 5px rgba(0,0,0,0.1);"
)


def _process_smarts_matches(
    mol: Any,
    smarts_mols: list[tuple[str, str, Any, tuple[float, float, float]]],
    match_counter: defaultdict | None,
) -> tuple[list[int], dict[int, tuple[float, float, float]], list[str]]:
    """Process SMARTS patterns and return atom highlights."""
    atom_ids: list[int] = []
    colors: dict[int, tuple[float, float, float]] = {}
    tooltips: list[str] = []

    for s_name, _, smarts_mol, color in smarts_mols:
        matches = mol.GetSubstructMatches(smarts_mol)
        if not matches:
            continue

        match_atoms = [idx for match in matches for idx in match]
        atom_ids.extend(match_atoms)
        colors.update({idx: color for idx in match_atoms})
        tooltips.append(f"✅ {s_name}: {len(matches)} match(es)")

        if match_counter is not None:
            match_counter[s_name] += 1

    return atom_ids, colors, tooltips


def _create_label(name: str, smi: str) -> str:
    """Create label for molecule display."""
    if name != smi:
        return f"<strong>{name}</strong><br><code>{smi}</code>"
    return f"<code>{smi}</code>"


def render_with_highlights(
    name: str,
    smi: str,
    smarts_mols: list[tuple[str, str, Any, tuple[float, float, float]]],
    match_counter: defaultdict | None = None,
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
    """
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DSVG
    from rdkit.Chem.rdDepictor import Compute2DCoords

    mol = MolFromSmiles(smi)
    if not mol:
        return _INVALID_SMILES_TEMPLATE.format(smi=smi)

    Compute2DCoords(mol)

    atom_ids, colors, tooltips = _process_smarts_matches(
        mol=mol,
        smarts_mols=smarts_mols,
        match_counter=match_counter,
    )

    drawer = MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol, highlightAtoms=atom_ids, highlightAtomColors=colors)
    drawer.FinishDrawing()

    label = _create_label(name=name, smi=smi)
    tooltips_html = "<br>".join(tooltips)

    return (
        f"<div style='{_SVG_CONTAINER_STYLE}'>"
        f"{drawer.GetDrawingText()}<br>{label}<br>"
        f"<small>{tooltips_html}</small></div>"
    )
