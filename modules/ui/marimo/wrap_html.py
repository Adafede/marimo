"""Wrap HTML string in mo.Html."""

__all__ = ["wrap_html"]

import marimo as mo


def wrap_html(html_str: str | None) -> mo.Html:
    """Wrap HTML string in mo.Html for mo.ui.table.

Parameters
----------
html_str : str | None
    Html str.

Returns
-------
mo.Html
    Computed result.
    """
    return mo.Html(html_str) if html_str else mo.Html("")
