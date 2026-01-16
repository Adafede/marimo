"""Select best taxon match from candidates."""

__all__ = ["select_best"]

from typing import Any, Dict, List, Optional, Tuple


def select_best(
    matches: List[Tuple[str, str, Optional[str], Optional[str], Optional[int]]],
    connectivity_map: Dict[str, int],
    original_name: str,
) -> Tuple[Optional[str], bool, List[Any]]:
    """
    Select the best matching taxon from a list of candidates.
    
    Returns:
        Tuple of:
        - Selected QID (or None if no matches)
        - Whether it was an exact match
        - List of matches for disambiguation display
    """
    if not matches:
        return None, False, []

    name_lower = original_name.lower().strip()
    exact_matches = [m for m in matches if m[1].lower() == name_lower]

    if exact_matches:
        sorted_matches = sorted(
            exact_matches,
            key=lambda x: connectivity_map.get(x[0], 0),
            reverse=True
        )
        return sorted_matches[0][0], True, sorted_matches

    sorted_matches = sorted(
        matches[:10],
        key=lambda x: connectivity_map.get(x[0], 0),
        reverse=True
    )
    return sorted_matches[0][0] if sorted_matches else None, False, sorted_matches
