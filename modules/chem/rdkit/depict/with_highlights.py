"""Depict molecule with highlighted substructures."""

__all__ = ["with_highlights"]

from collections import defaultdict

from ..mol.compute_2d_coords import compute_2d_coords
from ..smiles.parse import parse
from .collect_highlights import SmartsEntry, collect_highlights
from .svg_from_mol import svg_from_mol

INVALID_SMILES_TEMPLATE: str = (
    "<div style='color:red;'>[x] Invalid SMILES: <code>{smi}</code></div>"
)

CONTAINER_STYLE: str = (
    "display:inline-block; margin:12px; text-align:center; "
    "border:1px solid #eee; padding:10px; border-radius:8px; "
    "box-shadow: 2px 2px 5px rgba(0,0,0,0.1);"
)


def create_label(name: str, smiles: str) -> str:
    """Create HTML label for molecule display.

    Parameters
    ----------
    name : str
        Name.
    smiles : str
        Smiles.

    Returns
    -------
    str
        String representation of label.
    """
    if name != smiles:
        return f"<strong>{name}</strong><br><code>{smiles}</code>"
    return f"<code>{smiles}</code>"


def wrap_in_container(svg: str, label: str, tooltips: list[str]) -> str:
    """Wrap SVG and metadata in styled HTML container.

    Parameters
    ----------
    svg : str
        Svg.
    label : str
        Label.
    tooltips : list[str]
        Tooltips.

    Returns
    -------
    str
        String representation of wrap in container.
    """
    tooltips_html = "<br>".join(tooltips)
    return (
        f"<div style='{CONTAINER_STYLE}'>"
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
    """Depict a molecule as SVG with highlighted substructures.

    Parameters
    ----------
    name : str
        Name.
    smiles : str
        Smiles.
    smarts_entries : list[SmartsEntry]
        Smarts entries.
    match_counter : defaultdict | None
        None. Default is None.
    width : int
        Default is 200.
    height : int
        Default is 200.

    Returns
    -------
    str
        String representation of with highlights.
    """
    mol = parse(smiles=smiles)
    if not mol:
        return INVALID_SMILES_TEMPLATE.format(smi=smiles)

    compute_2d_coords(mol=mol)

    atom_ids, colors, tooltips = collect_highlights(
        mol=mol,
        smarts_entries=smarts_entries,
        match_counter=match_counter,
    )

    svg = svg_from_mol(
        mol=mol,
        width=width,
        height=height,
        highlight_atoms=atom_ids,
        highlight_colors=colors,
    )

    label = create_label(name=name, smiles=smiles)

    return wrap_in_container(svg=svg, label=label, tooltips=tooltips)
