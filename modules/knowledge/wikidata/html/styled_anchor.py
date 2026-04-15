"""HTML link styling helper."""

__all__ = ["styled_anchor"]

DEFAULT_LINK_COLOR = "#3377c4"


def styled_anchor(url: str, text: str, color: str = DEFAULT_LINK_COLOR) -> str:
    """Create styled anchor tag with target=_blank.

    Parameters
    ----------
    url : str
        Url.
    text : str
        Text.
    color : str
        DEFAULT_LINK_COLOR. Default is DEFAULT_LINK_COLOR.

    Returns
    -------
    str
        String representation of styled anchor.
    """
    return f'<a href="{url}" target="_blank" style="color:{color};">{text}</a>'
