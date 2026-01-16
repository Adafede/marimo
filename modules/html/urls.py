"""
URL building utilities - no external dependencies.

Functions for constructing URLs and query strings.
"""

__all__ = [
    "build_query_string",
    "structure_image_url",
    "doi_url",
    "scholia_url",
]

from urllib.parse import urlencode, quote


def build_query_string(params: dict[str, str | int | float | bool]) -> str:
    """
    Build a URL query string from parameters.

    Args:
        params: Dictionary of parameter names to values.
                None values are excluded.
                Bool values become "true"/"false".

    Returns:
        Query string starting with "?" or empty string if no params

    Example:
        >>> build_query_string({"name": "test", "count": 5})
        '?name=test&count=5'
        >>> build_query_string({})
        ''
    """
    filtered = {}
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, bool):
            filtered[k] = "true" if v else "false"
        else:
            filtered[k] = str(v)

    if not filtered:
        return ""

    return "?" + urlencode(filtered)


def structure_image_url(
    smiles: str,
    base_url: str = "https://www.simolecule.com/cdkdepict/depict/cow/svg",
    annotate: str = "cip",
) -> str:
    """
    Generate URL for chemical structure image from SMILES.

    Args:
        smiles: SMILES string
        base_url: Base URL for the depiction service
        annotate: Annotation style (e.g., "cip" for stereochemistry)

    Returns:
        Complete URL for structure image

    Example:
        >>> structure_image_url("c1ccccc1")
        'https://www.simolecule.com/cdkdepict/depict/cow/svg?smi=c1ccccc1&annotate=cip'
    """
    encoded_smiles = quote(smiles)
    return f"{base_url}?smi={encoded_smiles}&annotate={annotate}"


def doi_url(doi: str) -> str:
    """
    Generate DOI resolver URL.

    Args:
        doi: DOI string (with or without URL prefix)

    Returns:
        Full DOI URL

    Example:
        >>> doi_url("10.1234/example")
        'https://doi.org/10.1234/example'
    """
    # Strip any existing URL prefix
    if doi.startswith("http"):
        doi = doi.split("doi.org/")[-1]
    return f"https://doi.org/{doi}"


def scholia_url(qid: str) -> str:
    """
    Generate Scholia URL for a Wikidata QID.

    Args:
        qid: Wikidata QID (e.g., "Q12345")

    Returns:
        Scholia URL

    Example:
        >>> scholia_url("Q12345")
        'https://scholia.toolforge.org/Q12345'
    """
    return f"https://scholia.toolforge.org/{qid}"
