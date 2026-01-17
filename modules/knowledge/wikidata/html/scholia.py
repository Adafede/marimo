"""Scholia link constants and URL builder."""

__all__ = ["SCHOLIA_BASE", "scholia_url"]

SCHOLIA_BASE = "https://scholia.toolforge.org/"


def scholia_url(qid: str) -> str:
    """Build Scholia URL from QID."""
    return f"{SCHOLIA_BASE}{qid}"
