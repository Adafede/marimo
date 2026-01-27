"""Ranks utilities."""

__all__ = ["normalize_rank", "get_rank_label", "get_rank_order"]


def normalize_rank(rank: str) -> str:
    """Strip Wikidata IRI prefix and normalize to lowercase."""
    if rank is None:
        return ""
    rank_str = str(rank)
    # Strip common Wikidata prefixes
    for prefix in [
        "http://www.wikidata.org/entity/",
        "https://www.wikidata.org/entity/",
        "wd:",
    ]:
        if rank_str.startswith(prefix):
            rank_str = rank_str[len(prefix) :]
            break
    return rank_str.lower()


def get_rank_label(rank: str) -> str:
    """Convert rank identifier (QID or name) to display label."""
    if rank is None:
        return "Unknown"
    rank_normalized = normalize_rank(rank)
    # Handle various formats: QIDs, names, mixed case
    labels = {
        # Wikidata QIDs
        "q22666877": "Superdomain",
        "q146481": "Domain",
        # "q3491996": "Subdomain",
        "q19858692": "Superkingdom",
        "q36732": "Kingdom",
        "q2752679": "Subkingdom",
        "q3150876": "Infrakingdom",
        # "q2111790": "Superphylum",
        "q38348": "Phylum",
        "q1153785": "Subphylum",
        # "q2361851": "Infraphylum",
        "q23760204": "Superdivision",
        "q334460": "Division",
        "q3491997": "Subdivision",
        # "q60922428": "Megaclass",
        # "q3504061": "Superclass",
        "q37517": "Class",
        "q5867051": "Subclass",
        # "q2007442": "Infraclass",
        # "q21061204": "Subterclass",
        # "q6054237": "Magnorder",
        # "q5868144": "Superorder",
        # "q6462265": "Grandorder",
        # "q7506274": "Mirorder",
        "q36602": "Order",
        # "q5867959": "Suborder",
        # "q2889003": "Infraorder",
        # "q6311258": "Parvorder",
        # "q2136103": "Superfamily",
        "q35409": "Family",
        "q164280": "Subfamily",
        # "q14817220": "Supertribe",
        "q227936": "Tribe",
        # "q3965313": "Subtribe",
        "q34740": "Genus",
        # "q3238261": "Subgenus",
        "q7432": "Species",
        # "q767728": "Variety",
        # Standard names
        "superdomain": "Superdomain",
        "domain": "Domain",
        # "subdomain": "Subdomain",
        "superkingdom": "Superkingdom",
        "kingdom": "Kingdom",
        "subkingdom": "Subkingdom",
        "infrakingdom": "Infrakingdom",
        # "superphylum": "Superphylum",
        "phylum": "Phylum",
        "subphylum": "Subphylum",
        # "infraphylum": "Infraphylum",
        "superdivision": "Superdivision",
        "division": "Division",
        "subdivision": "Subdivision",
        # "megaclass": "Megaclass",
        # "superclass": "Superclass",
        "class": "Class",
        "subclass": "Subclass",
        # "infraclass": "Infraclass",
        # "subterclass": "Subterclass",
        # "magnorder": "Magnorder",
        # "superorder": "Superorder",
        # "grandorder": "Grandorder",
        # "mirorder": "Mirorder",
        "order": "Order",
        # "suborder": "Suborder",
        # "infraorder": "Infraorder",
        # "parvorder": "Parvorder",
        # "superfamily": "Superfamily",
        "family": "Family",
        "subfamily": "Subfamily",
        # "supertribe": "Supertribe",
        "tribe": "Tribe",
        # "subtribe": "Subtribe",
        "genus": "Genus",
        # "subgenus": "Subgenus",
        "species": "Species",
        # "variety": "Variety",
    }
    return labels.get(
        rank_normalized,
        rank_normalized.upper()
        if rank_normalized.startswith("q")
        else rank_normalized.capitalize(),
    )


def get_rank_order(rank: str) -> int:
    """Get sort order for a rank (lower = coarser taxonomic level)."""
    if rank is None:
        return 999
    rank_normalized = normalize_rank(rank)
    order = {
        # Wikidata QIDs
        "q22666877": 0,
        "q146481": 1,
        # "q3491996": 2,
        "q19858692": 3,
        "q36732": 4,
        "q2752679": 5,
        "q3150876": 6,
        # "q2111790": 7,
        "q38348": 8,
        "q1153785": 9,
        # "q2361851": 10,
        "q23760204": 11,
        "q334460": 12,
        "q3491997": 13,
        # "q60922428": 14,
        # "q3504061": 15,
        "q37517": 16,
        "q5867051": 17,
        # "q2007442": 18,
        # "q21061204": 19,
        # "q6054237": 20,
        # "q5868144": 21,
        # "q6462265": 22,
        # "q7506274": 23,
        "q36602": 24,
        # "q5867959": 25,
        # "q2889003": 26,
        # "q6311258": 27,
        # "q2136103": 28,
        "q35409": 29,
        "q164280": 30,
        # "q14817220": 31,
        "q227936": 32,
        # "q3965313": 33,
        "q34740": 34,
        # "q3238261": 35,
        "q7432": 36,
        # "q767728": 37,
        # Standard names
        "superdomain": 0,
        "domain": 1,
        # "subdomain": 2,
        "superkingdom": 3,
        "kingdom": 4,
        "subkingdom": 5,
        "infrakingdom": 6,
        # "superphylum": 7,
        "phylum": 8,
        "subphylum": 9,
        # "infraphylum": 10,
        "superdivision": 11,
        "division": 12,
        "subdivision": 13,
        # "megaclass": 14,
        # "superclass": 15,
        "class": 16,
        "subclass": 17,
        # "infraclass": 18,
        # "subterclass": 19,
        # "magnorder": 20,
        # "superorder": 21,
        # "grandorder": 22,
        # "mirorder": 23,
        "order": 24,
        # "suborder": 25,
        # "infraorder": 26,
        # "parvorder": 27,
        # "superfamily": 28,
        "family": 29,
        "subfamily": 30,
        # "supertribe": 31,
        "tribe": 32,
        # "subtribe": 33,
        "genus": 34,
        # "subgenus": 35,
        "species": 36,
        # "variety": 37,
    }
    return order.get(rank_normalized, 999)
