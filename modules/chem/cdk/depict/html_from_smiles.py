"""Generate HTML img tag for CDK Depict from SMILES string."""

__all__ = ["html_from_smiles"]

from .url import CDK_DEPICT_URL
from .url_from_smiles import url_from_smiles

DEFAULT_MAX_WIDTH = "150px"
DEFAULT_MAX_HEIGHT = "100px"


def html_from_smiles(
    smiles: str,
    base_url: str = CDK_DEPICT_URL,
    layout: str = "cow",
    img_format: str = "svg",
    annotate: str | None = "cip",
    max_width: str = DEFAULT_MAX_WIDTH,
    max_height: str = DEFAULT_MAX_HEIGHT,
    rounded: bool = True,
    lazy: bool = True,
) -> str:
    """Generate HTML img tag for a chemical structure depiction from SMILES."""
    if not smiles:
        return ""

    img_url = url_from_smiles(
        smiles=smiles,
        base_url=base_url,
        layout=layout,
        img_format=img_format,
        annotate=annotate,
    )

    style_parts = [f"max-width:{max_width}", f"max-height:{max_height}"]
    if rounded:
        style_parts.append("border-radius:8px")

    loading = 'loading="lazy" ' if lazy else ""
    return f'<img src="{img_url}" {loading}style="{";".join(style_parts)};" />'
