"""Generate HTML img tag for CDK Depict from SMILES string."""

__all__ = ["html_from_smiles"]

from .url import CDK_DEPICT_URL
from .url_from_smiles import url_from_smiles

DEFAULT_MAX_WIDTH: str = "150px"
DEFAULT_MAX_HEIGHT: str = "100px"
DEFAULT_BORDER_RADIUS: str = "8px"


def _build_style(
    max_width: str,
    max_height: str,
    rounded: bool,
) -> str:
    """Build CSS style string for img tag."""
    styles = {
        "max-width": max_width,
        "max-height": max_height,
        **({"border-radius": DEFAULT_BORDER_RADIUS} if rounded else {}),
    }
    return ";".join(f"{k}:{v}" for k, v in styles.items())


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

    style = _build_style(max_width=max_width, max_height=max_height, rounded=rounded)
    loading_attr = 'loading="lazy" ' if lazy else ""

    return f'<img src="{img_url}" {loading_attr}style="{style};" />'
