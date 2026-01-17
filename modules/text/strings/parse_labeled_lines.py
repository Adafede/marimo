"""Parse labeled lines."""

__all__ = ["parse_labeled_lines"]


def parse_line(line: str) -> tuple[str, str]:
    """Parse a single line into (label, value) tuple."""
    if " " in line:
        value, label = line.split(sep=" ", maxsplit=1)
        return label.strip(), value.strip()
    return line, line


def parse_labeled_lines(text: str) -> list[tuple[str, str]]:
    """Parse lines with optional labels (value label format)."""
    return [
        parse_line(line=line.strip()) for line in text.splitlines() if line.strip()
    ]
