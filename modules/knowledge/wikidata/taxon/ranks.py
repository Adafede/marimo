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
        "22666877": "Superdomain",
        "146481": "Domain",
        # "3491996": "Subdomain",
        "19858692": "Superkingdom",
        "36732": "Kingdom",
        "2752679": "Subkingdom",
        "3150876": "Infrakingdom",
        # "2111790": "Superphylum",
        "38348": "Phylum",
        "1153785": "Subphylum",
        # "2361851": "Infraphylum",
        "23760204": "Superdivision",
        "334460": "Division",
        "3491997": "Subdivision",
        # "60922428": "Megaclass",
        # "3504061": "Superclass",
        "37517": "Class",
        "5867051": "Subclass",
        # "2007442": "Infraclass",
        # "21061204": "Subterclass",
        # "6054237": "Magnorder",
        # "5868144": "Superorder",
        # "6462265": "Grandorder",
        # "7506274": "Mirorder",
        "36602": "Order",
        # "5867959": "Suborder",
        # "2889003": "Infraorder",
        # "6311258": "Parvorder",
        # "2136103": "Superfamily",
        "35409": "Family",
        "164280": "Subfamily",
        # "14817220": "Supertribe",
        "227936": "Tribe",
        # "3965313": "Subtribe",
        "34740": "Genus",
        # "3238261": "Subgenus",
        "7432": "Species",
        # "767728": "Variety",
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
        if rank_normalized.startswith("")
        else rank_normalized.capitalize(),
    )


def get_rank_order(rank: str) -> int:
    """Get sort order for a rank (lower = coarser taxonomic level)."""
    if rank is None:
        return 999
    rank_normalized = normalize_rank(rank)
    order = {
        # Wikidata QIDs
        "22666877": 0,
        "146481": 1,
        # "3491996": 2,
        "19858692": 3,
        "36732": 4,
        "2752679": 5,
        "3150876": 6,
        # "2111790": 7,
        "38348": 8,
        "1153785": 9,
        # "2361851": 10,
        "23760204": 11,
        "334460": 12,
        "3491997": 13,
        # "60922428": 14,
        # "3504061": 15,
        "37517": 16,
        "5867051": 17,
        # "2007442": 18,
        # "21061204": 19,
        # "6054237": 20,
        # "5868144": 21,
        # "6462265": 22,
        # "7506274": 23,
        "36602": 24,
        # "5867959": 25,
        # "2889003": 26,
        # "6311258": 27,
        # "2136103": 28,
        "35409": 29,
        "164280": 30,
        # "14817220": 31,
        "227936": 32,
        # "3965313": 33,
        "34740": 34,
        # "3238261": 35,
        "7432": 36,
        # "767728": 37,
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
