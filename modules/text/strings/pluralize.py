"""String pluralization."""

__all__ = ["pluralize"]


def pluralize(
    singular: str, count: int, irregular: dict[str, str] | None = None
) -> str:
    """Return singular or plural form based on count."""
    if count == 1:
        return singular
    if irregular and singular in irregular:
        return irregular[singular]
    return f"{singular}s"
