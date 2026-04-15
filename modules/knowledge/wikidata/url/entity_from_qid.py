"""Build Wikidata entity URL from QID."""

__all__ = ["entity_from_qid"]

from .constants import ENTITY_PREFIX


def entity_from_qid(qid: str) -> str:
    """Build Wikidata entity URL for a QID.

    Parameters
    ----------
    qid : str
        Qid.

    Returns
    -------
    str
        Return value produced by entity from qid.
    """
    return f"{ENTITY_PREFIX}{qid}"
