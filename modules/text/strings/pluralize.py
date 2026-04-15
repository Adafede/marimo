"""String pluralization."""

__all__ = ["pluralize"]


def pluralize(
    singular: str,
    count: int,
    irregular: dict[str, str] | None = None,
) -> str:
    """Return singular or plural form based on count.

    Parameters
    ----------
    singular : str
        Singular.
    count : int
        Count.
    irregular : dict[str, str] | None
        None. Default is None.

    Returns
    -------
    str
        Return value produced by pluralize.
    """
    if count == 1:
        return singular
    if irregular and singular in irregular:
        return irregular[singular]
    return f"{singular}s"
