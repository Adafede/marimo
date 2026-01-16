"""HTML link styling helper."""

__all__ = ["styled_anchor"]

DEFAULT_LINK_COLOR = "#3377c4"


def styled_anchor(url: str, text: str, color: str = DEFAULT_LINK_COLOR) -> str:
    """Create styled anchor tag with target=_blank."""
    return f'<a href="{url}" target="_blank" style="color:{color};">{text}</a>'
