"""Depict molecule with highlighted substructures."""

__all__ = ["with_highlights"]

from collections import defaultdict

from ..mol.compute_2d_coords import compute_2d_coords
from ..smiles.parse import parse as parse_smiles
from .collect_highlights import SmartsEntry, collect_highlights
from .to_svg import to_svg

_INVALID_SMILES_TEMPLATE: str = (
    "<div style='color:red;'>[x] Invalid SMILES: <code>{smi}</code></div>"
)

_CONTAINER_STYLE: str = (
    "display:inline-block; margin:12px; text-align:center; "
    "border:1px solid #eee; padding:10px; border-radius:8px; "
    "box-shadow: 2px 2px 5px rgba(0,0,0,0.1);"
)


def create_label(name: str, smiles: str) -> str:
    """Create HTML label for molecule display."""
    if name != smiles:
        return f"<strong>{name}</strong><br><code>{smiles}</code>"
    return f"<code>{smiles}</code>"


def wrap_in_container(svg: str, label: str, tooltips: list[str]) -> str:
    """Wrap SVG and metadata in styled HTML container."""
    tooltips_html = "<br>".join(tooltips)
    return (
        f"<div style='{_CONTAINER_STYLE}'>"
        f"{svg}<br>{label}<br>"
        f"<small>{tooltips_html}</small></div>"
    )


def with_highlights(
    name: str,
    smiles: str,
    smarts_entries: list[SmartsEntry],
    match_counter: defaultdict | None = None,
    width: int = 200,
    height: int = 200,
) -> str:
    """
    Depict a molecule as SVG with highlighted substructures.

    Args:
        name: Display name for the molecule
        smiles: SMILES string
        smarts_entries: List of (name, smarts, smarts_mol, rgb_color) tuples
        match_counter: Optional defaultdict to count matches per SMARTS
        width: SVG width in pixels
        height: SVG height in pixels

    Returns:
        HTML string with SVG and labels
    """
    mol = parse_smiles(smiles=smiles)
    if not mol:
        return _INVALID_SMILES_TEMPLATE.format(smi=smiles)

    compute_2d_coords(mol=mol)

    atom_ids, colors, tooltips = collect_highlights(
        mol=mol,
        smarts_entries=smarts_entries,
        match_counter=match_counter,
    )

    svg = to_svg(
        mol=mol,
        width=width,
        height=height,
        highlight_atoms=atom_ids,
        highlight_colors=colors,
    )

    label = create_label(name=name, smiles=smiles)

    return wrap_in_container(svg=svg, label=label, tooltips=tooltips)
