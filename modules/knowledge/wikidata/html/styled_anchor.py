"""HTML link styling helper."""

__all__ = ["styled_anchor"]


def styled_anchor(url: str, text: str, color: str = "#3377c4") -> str:
    """Create styled anchor tag with target=_blank."""
    return f'<a href="{url}" target="_blank" style="color:{color};">{text}</a>'
