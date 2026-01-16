"""Parse labeled lines."""

__all__ = ["parse_labeled_lines"]


def parse_labeled_lines(text: str) -> list[tuple[str, str]]:
    """Parse lines with optional labels (value label format)."""
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
