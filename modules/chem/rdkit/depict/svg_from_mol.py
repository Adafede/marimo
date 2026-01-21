"""Draw molecule to SVG string."""

__all__ = ["svg_from_mol"]

from rdkit.Chem import Mol
from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DSVG

DEFAULT_WIDTH: int = 200
DEFAULT_HEIGHT: int = 200


def svg_from_mol(
    mol: Mol,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    highlight_atoms: list[int] | None = None,
    highlight_colors: dict[int, tuple[float, float, float]] | None = None,
) -> str:
    """
    Draw molecule to SVG string.

    Args:
        mol: RDKit Mol object
        width: SVG width in pixels
        height: SVG height in pixels
        highlight_atoms: List of atom indices to highlight
        highlight_colors: Dict mapping atom index to RGB color tuple (0.0-1.0)

    Returns:
        SVG string
    """
    drawer = MolDraw2DSVG(width, height)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms or [],
        highlightAtomColors=highlight_colors or {},
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
