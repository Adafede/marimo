"""Render molecule as SVG with highlighted substructures."""

__all__ = ["render_with_highlights"]

from collections import defaultdict
from typing import Any


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
