"""Generate molecule depiction data URI from Indigo molecule."""

__all__ = ["svg_from_mol"]

import base64
from indigo import IndigoObject
from ..indigo_instance import get_indigo, get_renderer


def svg_from_mol(
    mol: IndigoObject | None,
    img_format: str = "svg",
    aromatize: bool = True,
    coloring: bool = True,
) -> str:
    """Generate data URI for chemical structure depiction from an Indigo molecule.

Parameters
----------
mol : IndigoObject | None
    Mol.
img_format : str
    Default is 'svg'.
aromatize : bool
    True. Default is True.
coloring : bool
    True. Default is True.

Returns
-------
str
    Computed result.
    """
    if mol is None:
        return ""

    indigo = get_indigo()
    renderer = get_renderer()

    try:
        indigo.setOption("render-output-format", img_format.lower())
        indigo.setOption("render-coloring", coloring)

        if aromatize:
            mol.aromatize()

        mol.layout()  # REQUIRED before rendering

        rendered = renderer.renderToString(mol)

        if img_format.lower() == "svg":
            mime_type = "image/svg+xml"
            encoded = base64.b64encode(rendered.encode()).decode()
        else:
            mime_type = f"image/{img_format.lower()}"
            encoded = base64.b64encode(rendered).decode()

        return f"data:{mime_type};base64,{encoded}"

    except Exception:
        return ""
