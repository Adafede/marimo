"""Wrap HTML string in mo.Html."""

__all__ = ["wrap_html"]

import marimo as mo


def wrap_html(html_str: str) -> mo.Html:
    """Wrap HTML string in mo.Html for mo.ui.table."""
    return mo.Html(html_str) if html_str else mo.Html("")
