"""Build Wikidata wiki URL from QID."""

__all__ = ["wiki_from_qid"]

from .constants import WIKI_PREFIX


def wiki_from_qid(qid: str) -> str:
    """Build Wikidata wiki URL for a QID.

    Parameters
    ----------
    qid : str
        Qid.

    Returns
    -------
    str
        String representation of wiki from qid.

    """
    return f"{WIKI_PREFIX}{qid}"
