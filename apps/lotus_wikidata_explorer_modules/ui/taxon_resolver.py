"""
Taxon resolution logic - convert taxon names to Wikidata QIDs.

Handles ambiguity, fuzzy matching, and compound count-based ranking.
"""

from typing import Optional, Tuple, List, Dict, Any
import marimo as mo

from ..core.config import CONFIG, SCHOLIA_URL
from ..core.utils import extract_qid
from ..query.builders import build_sparql_values_clause
from ..query.executor import execute_sparql
from ..core.constants import SPARQL_PREFIXES

__all__ = [
    "resolve_taxon_to_qid",
    "create_taxon_warning_html",
]


def _build_taxon_search_query(taxon_name: str) -> str:
    """Build SPARQL query to find taxa by scientific name."""
    return f"""
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT ?taxon ?taxon_name WHERE {{
        VALUES ?taxon_name {{ "{taxon_name}" }}
        ?taxon wdt:P225 ?taxon_name .
    }}
    """


def _build_taxon_details_query(qids: List[str]) -> str:
    """Build query to fetch taxon details (labels, descriptions, parents)."""
    values_clause = build_sparql_values_clause("taxon", qids)
    return f"""
    {SPARQL_PREFIXES}

    SELECT ?taxon ?taxonLabel ?taxonDescription ?taxon_parent ?taxon_parentLabel 
    WHERE {{
      {values_clause}

      # Parent taxon
      OPTIONAL {{ ?taxon wdt:P171 ?taxon_parent }}

      # Taxon labels (English preferred)
      OPTIONAL {{
        ?taxon rdfs:label ?taxonLabel .
        FILTER(LANG(?taxonLabel) = "en")
      }}
      OPTIONAL {{
        ?taxon rdfs:label ?taxonLabel .
        FILTER(LANG(?taxonLabel) = "mul")
      }}

      # Taxon descriptions
      OPTIONAL {{
        ?taxon schema:description ?taxonDescription .
        FILTER(LANG(?taxonDescription) = "en")
      }}
      OPTIONAL {{
        ?taxon schema:description ?taxonDescription .
        FILTER(LANG(?taxonDescription) = "mul")
      }}

      # Parent labels
      OPTIONAL {{
        ?taxon_parent rdfs:label ?taxon_parentLabel .
        FILTER(LANG(?taxon_parentLabel) = "en")
      }}
      OPTIONAL {{
        ?taxon_parent rdfs:label ?taxon_parentLabel .
        FILTER(LANG(?taxon_parentLabel) = "mul")
      }}
    }}
    """


def _build_taxon_connectivity_query(qids: List[str]) -> str:
    """Build query to count compound connections for each taxon."""
    values_clause = build_sparql_values_clause("taxon", qids)
    return f"""
    {SPARQL_PREFIXES}

    SELECT ?taxon (COUNT(DISTINCT ?compound) AS ?compound_count) WHERE {{
      {values_clause}

      # Count compounds directly linked to this taxon or its descendants
      {{
        SELECT ?taxon ?compound WHERE {{
          {values_clause}
          ?descendant (wdt:P171*) ?taxon .
          ?compound wdt:P235 ?inchikey ;
                    p:P703/ps:P703 ?descendant .
        }}
      }}
    }}
    GROUP BY ?taxon
    ORDER BY DESC(?compound_count)
    """


def _get_binding_value(binding: Dict[str, Any], key: str, default: str = "") -> str:
    """Extract value from SPARQL binding."""
    return binding.get(key, {}).get("value", default)


def create_taxon_warning_html(
    matches: list, selected_qid: str, is_exact: bool
) -> mo.Html:
    """
    Create an HTML warning with clickable QID links and taxon details.

    Args:
        matches: List of tuples (qid, name, description, parent, compound_count)
        selected_qid: QID that was selected
        is_exact: Whether matches are exact or fuzzy

    Returns:
        marimo Html object with formatted warning
    """
    match_type = "exact matches" if is_exact else "similar taxa"
    intro = (
        f"Ambiguous taxon name. Multiple {match_type} found:"
        if is_exact
        else f"No exact match. Similar taxa found:"
    )

    # Build HTML list of matches
    items = []
    for match_data in matches:
        qid = match_data[0]
        name = match_data[1]
        description = match_data[2] if len(match_data) > 2 else None
        parent = match_data[3] if len(match_data) > 3 else None
        compound_count = match_data[4] if len(match_data) > 4 else None

        # Create clickable link
        link = f'<a href="{SCHOLIA_URL}{qid}" target="_blank" rel="noopener noreferrer" style="color: {CONFIG["color_hyperlink"]}; text-decoration: none; border-bottom: 1px solid transparent; font-weight: bold;">{qid}</a>'

        # Build details string
        details = []
        if name:
            details.append(f"<em>{name}</em>")
        if description:
            details.append(f"{description}")
        if parent:
            details.append(f"parent: {parent}")
        if compound_count is not None:
            details.append(f"<strong>{compound_count:,} compounds</strong>")

        details_str = " - ".join(details) if details else ""

        # Highlight the selected one
        if qid == selected_qid:
            items.append(
                f"<li>{link} {details_str} <strong>← USING THIS ONE (most compounds)</strong></li>"
            )
        else:
            items.append(f"<li>{link} {details_str}</li>")

    items_html = "".join(items)

    html = f"""
    <div style="line-height: 1.6;">
        {intro}
        <ul style="margin: 0.5em 0; padding-left: 1.5em;">
            {items_html}
        </ul>
        <em>For precision, please use a specific QID directly in the search box. When ambiguous, the taxon with the most compound links is automatically selected.</em>
    </div>
    """

    return mo.Html(html)


def resolve_taxon_to_qid(taxon_input: str) -> Tuple[Optional[str], Optional[mo.Html]]:
    """
    Resolve taxon name or QID to a valid QID.

    Args:
        taxon_input: Taxon name or Wikidata QID

    Returns:
        Tuple of (qid, warning_html) where warning_html is present if ambiguous
    """
    taxon_input = taxon_input.strip()

    # Handle wildcard for all taxa
    if taxon_input == "*":
        return "*", None

    # Early return if input is already a QID
    if taxon_input.upper().startswith("Q") and taxon_input[1:].isdigit():
        return taxon_input.upper(), None

    # Search for taxon by name
    try:
        query = _build_taxon_search_query(taxon_input)
        results = execute_sparql(query)
        bindings = results.get("results", {}).get("bindings", [])

        if not bindings:
            return None, None

        # Extract matches
        matches = [
            (extract_qid(b["taxon"]["value"]), b["taxon_name"]["value"])
            for b in bindings
            if "taxon" in b
            and "taxon_name" in b
            and "value" in b["taxon"]
            and "value" in b["taxon_name"]
        ]

        if not matches:
            return None, None

        # Find exact matches (case-insensitive)
        taxon_lower = taxon_input.lower()
        exact_matches = [
            (qid, name) for qid, name in matches if name.lower() == taxon_lower
        ]

        # Single exact match - perfect
        if len(exact_matches) == 1:
            return exact_matches[0][0], None

        # Multiple exact matches or similar matches - need to fetch details
        if len(exact_matches) > 1:
            # Get details for exact matches
            qids = [qid for qid, _ in exact_matches]

            # Query connectivity to find the most connected taxon
            connectivity_query = _build_taxon_connectivity_query(qids)
            connectivity_results = execute_sparql(connectivity_query)
            connectivity_bindings = connectivity_results.get("results", {}).get(
                "bindings", []
            )

            # Build connectivity map
            connectivity_map = {}
            for b in connectivity_bindings:
                qid = extract_qid(_get_binding_value(b, "taxon"))
                count = int(_get_binding_value(b, "compound_count", "0"))
                connectivity_map[qid] = count

            # Sort exact matches by connectivity (descending)
            sorted_matches = sorted(
                exact_matches, key=lambda x: connectivity_map.get(x[0], 0), reverse=True
            )

            # Get details for display
            details_query = _build_taxon_details_query(qids)
            details_results = execute_sparql(details_query)
            details_bindings = details_results.get("results", {}).get("bindings", [])

            # Build a map of QID to details
            details_map = {}
            for b in details_bindings:
                qid = extract_qid(_get_binding_value(b, "taxon"))
                details_map[qid] = (
                    _get_binding_value(b, "taxonLabel"),
                    _get_binding_value(b, "taxonDescription"),
                    _get_binding_value(b, "taxon_parentLabel"),
                )

            # Create matches with details including connectivity
            matches_with_details = [
                (
                    qid,
                    name,
                    details_map.get(qid, ("", "", ""))[1],
                    details_map.get(qid, ("", "", ""))[2],
                    connectivity_map.get(qid, 0),
                )
                for qid, name in sorted_matches
            ]

            # Use the most connected taxon (first in sorted list)
            selected_qid = sorted_matches[0][0]
            warning_html = create_taxon_warning_html(
                matches_with_details, selected_qid, is_exact=True
            )
            return selected_qid, warning_html

        # No exact match - use first result with warning
        if len(matches) > 1:
            # Get details for similar matches (limit to 5)
            qids = [qid for qid, _ in matches[:5]]

            # Query connectivity to find the most connected taxon
            connectivity_query = _build_taxon_connectivity_query(qids)
            connectivity_results = execute_sparql(connectivity_query)
            connectivity_bindings = connectivity_results.get("results", {}).get(
                "bindings", []
            )

            # Build connectivity map
            connectivity_map = {}
            for b in connectivity_bindings:
                qid = extract_qid(_get_binding_value(b, "taxon"))
                count = int(_get_binding_value(b, "compound_count", "0"))
                connectivity_map[qid] = count

            # Sort matches by connectivity (descending)
            sorted_matches = sorted(
                matches[:5], key=lambda x: connectivity_map.get(x[0], 0), reverse=True
            )

            # Get details for display
            details_query = _build_taxon_details_query(qids)
            details_results = execute_sparql(details_query)
            details_bindings = details_results.get("results", {}).get("bindings", [])

            # Build a map of QID to details
            details_map = {}
            for b in details_bindings:
                qid = extract_qid(_get_binding_value(b, "taxon"))
                details_map[qid] = (
                    _get_binding_value(b, "taxonLabel"),
                    _get_binding_value(b, "taxonDescription"),
                    _get_binding_value(b, "taxon_parentLabel"),
                )

            # Create matches with details including connectivity
            matches_with_details = [
                (
                    qid,
                    name,
                    details_map.get(qid, ("", "", ""))[1],
                    details_map.get(qid, ("", "", ""))[2],
                    connectivity_map.get(qid, 0),
                )
                for qid, name in sorted_matches
            ]

            # Use the most connected taxon (first in sorted list)
            selected_qid = sorted_matches[0][0]
            warning_html = create_taxon_warning_html(
                matches_with_details, selected_qid, is_exact=False
            )
            return selected_qid, warning_html

        return matches[0][0], None

    except Exception:
        return None, None
