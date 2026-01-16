"""Lazy structure image for mo.Html."""

__all__ = ["lazy_structure_image"]

from collections.abc import Callable

import marimo as mo


def lazy_structure_image(
    smiles: str | None,
    html_generator_func: Callable[..., str],
    base_url: str = "https://www.simolecule.com/cdkdepict",
    max_width: str = "200px",
    max_height: str = "150px",
) -> mo.Html:
    """Create a lazy-loading structure image wrapped in mo.Html."""
    if not smiles:
        return mo.Html("")

    html = html_generator_func(
        smiles, base_url, max_width=max_width, max_height=max_height
    )
    return mo.Html(html) if html else mo.Html("")
