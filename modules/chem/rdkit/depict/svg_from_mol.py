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
    """Draw molecule to SVG string.

    Parameters
    ----------
    mol : Mol
        Mol.
    width : int
        DEFAULT_WIDTH. Default is DEFAULT_WIDTH.
    height : int
        DEFAULT_HEIGHT. Default is DEFAULT_HEIGHT.
    highlight_atoms : list[int] | None
        None. Default is None.
    highlight_colors : dict[int, tuple[float, float, float]] | None
        None. Default is None.

    Returns
    -------
    str
        Return value produced by svg from mol.
    """
    drawer = MolDraw2DSVG(width, height)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms or [],
        highlightAtomColors=highlight_colors or {},
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
