"""Select best taxon match from candidates."""

__all__ = ["select_best"]

from typing import Any


def select_best(
    matches: list[tuple[str, str, str | None, str | None, int | None]],
    connectivity_map: dict[str, int],
    original_name: str,
) -> tuple[str | None, bool, list[Any]]:
    """Select the best matching taxon from a list of candidates.

Parameters
----------
matches : list[tuple[str, str, str | None, str | None, int | None]]
    Matches.
connectivity_map : dict[str, int]
    Connectivity map.
original_name : str
    Original name.

Returns
-------
tuple[str | None, bool, list[Any]]
    Computed result.
    """
    if not matches:
        return None, False, []

    name_lower = original_name.lower().strip()
    exact_matches = [m for m in matches if m[1].lower() == name_lower]

    if exact_matches:
        sorted_matches = sorted(
            exact_matches,
            key=lambda x: connectivity_map.get(x[0], 0),
            reverse=True,
        )
        return sorted_matches[0][0], True, sorted_matches

    sorted_matches = sorted(
        matches[:10],
        key=lambda x: connectivity_map.get(x[0], 0),
        reverse=True,
    )
    return sorted_matches[0][0] if sorted_matches else None, False, sorted_matches
