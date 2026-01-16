"""
Pure text utilities - no external dependencies.

Functions for string manipulation, pluralization, validation, etc.
"""

__all__ = [
    "pluralize",
    "truncate",
    "parse_labeled_lines",
]


def pluralize(
    singular: str,
    count: int,
    irregular: dict[str, str] | None = None,
) -> str:
    """
    Return singular or plural form based on count.

    Args:
        singular: The singular form of the word
        count: The count to determine plural
        irregular: Optional dict of irregular plurals {singular: plural}

    Returns:
        Singular if count == 1, else plural form

    Example:
        >>> pluralize("Compound", 1)
        'Compound'
        >>> pluralize("Compound", 5)
        'Compounds'
        >>> pluralize("Taxon", 5, {"Taxon": "Taxa"})
        'Taxa'
    """
    if count == 1:
        return singular
    if irregular and singular in irregular:
        return irregular[singular]
    return f"{singular}s"


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to max_length, adding suffix if truncated.

    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add if truncated

    Returns:
        Truncated text with suffix, or original if short enough

    Example:
        >>> truncate("Hello World", 8)
        'Hello...'
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def parse_labeled_lines(text: str) -> list[tuple[str, str]]:
    """
    Parse lines with optional labels (value label format).

    Each non-empty line is parsed as "value label" or just "value".
    If no label is provided, the value is used as the label.

    Args:
        text: Multi-line text to parse

    Returns:
        List of (label, value) tuples

    Example:
        >>> parse_labeled_lines("CCO Ethanol\\nC1CCCCC1")
        [('Ethanol', 'CCO'), ('C1CCCCC1', 'C1CCCCC1')]
    """
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if " " in line:
            val, name = line.split(" ", 1)
            items.append((name.strip(), val.strip()))
        else:
            items.append((line, line))
    return items
