"""Generate CDK Depict URL from SMILES string."""

__all__ = ["url_from_smiles"]

from urllib.parse import quote

from .url import CDK_DEPICT_URL


def url_from_smiles(
    smiles: str | None,
    base_url: str = CDK_DEPICT_URL,
    layout: str = "cow",
    img_format: str = "svg",
    annotate: str | None = "cip",
) -> str:
    """Generate URL for chemical structure depiction from SMILES."""
    if not smiles:
        return ""
    url = f"{base_url}/depict/{layout}/{img_format}?smi={quote(string=smiles)}"
    if annotate is not None:
        url += f"&annotate={annotate}"
    return url
