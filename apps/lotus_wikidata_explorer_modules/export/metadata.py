"""
FAIR-compliant metadata generation for exported datasets.

Provides cryptographic provenance tracking and Schema.org-compliant metadata.
"""

from datetime import datetime
from typing import Dict, Any, Optional
import hashlib
import json
import polars as pl

from ..core.config import CONFIG, WIKIDATA_WIKI_URL, WIKIDATA_URL

__all__ = [
    "create_export_metadata",
    "create_dataset_hashes",
    "create_citation_text",
]


def create_dataset_hashes(
    qid: str,
    taxon_input: str,
    filters: Optional[Dict[str, Any]],
    df: pl.DataFrame,
) -> tuple[str, str]:
    """
    Create provenance hashes for reproducibility.

    Query hash: based on search parameters (what was asked)
    Result hash: based on compound identifiers (what was found)

    Args:
        qid: Wikidata QID of taxon
        taxon_input: Original taxon input string
        filters: Active filters dictionary
        df: Results dataframe

    Returns:
        Tuple of (query_hash, result_hash)
    """
    # Query hash - based on search parameters (what was asked)
    query_components = [qid or "", taxon_input or ""]
    if filters:
        query_components.append(json.dumps(filters, sort_keys=True))

    query_hash = hashlib.sha256("|".join(query_components).encode("utf-8")).hexdigest()

    # Result hash - based on actual compound identifiers (what was found)
    compound_ids = sorted(
        [
            row.get("compound_qid", "")
            for row in df.iter_rows(named=True)
            if row.get("compound_qid")
        ]
    )
    result_hash = hashlib.sha256("|".join(compound_ids).encode("utf-8")).hexdigest()

    return query_hash, result_hash


def create_export_metadata(
    df: pl.DataFrame,
    taxon_input: str,
    qid: str,
    filters: Dict[str, Any],
    query_hash: Optional[str] = None,
    result_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create FAIR-compliant metadata for exported datasets.

    Args:
        df: Results dataframe
        taxon_input: Original taxon input string
        qid: Wikidata QID of taxon
        filters: Active filters dictionary
        query_hash: SHA-256 hash of query parameters (optional)
        result_hash: SHA-256 hash of result identifiers (optional)

    Returns:
        Schema.org-compliant metadata dictionary
    """
    # Build descriptive name and description based on search type
    smiles_info = filters.get("chemical_structure", {}) if filters else {}

    if smiles_info:
        search_type = smiles_info.get("search_type", "substructure")
        smiles_str = smiles_info.get("smiles", "")

        if qid:
            # Combined search
            dataset_name = f"LOTUS Data - {search_type.title()} search in {taxon_input}"
            description = (
                f"Chemical compounds matching {search_type} search "
                f"(SMILES: {smiles_str}) "
                f"within taxon {taxon_input} (Wikidata QID: {qid}). "
            )
        else:
            # SMILES-only search
            dataset_name = f"LOTUS Data - Chemical {search_type.title()} Search"
            description = (
                f"Chemical compounds matching {search_type} search "
                f"(SMILES: {smiles_str}). "
            )

        if search_type == "similarity":
            threshold = smiles_info.get("similarity_threshold", 0.8)
            description += f"Tanimoto similarity threshold: {threshold}. "
    else:
        # Taxon-only search
        dataset_name = f"LOTUS Data - {taxon_input}"
        description = f"Chemical compounds from taxon {taxon_input} " + (
            f"(Wikidata QID: {qid}). " if qid else ". "
        )

    description += "Retrieved via LOTUS Wikidata Explorer with chemical search capabilities (SACHEM/IDSM)."

    metadata = {
        "@context": "https://schema.org/",
        "@type": "Dataset",
        "name": dataset_name,
        "description": description,
        "version": CONFIG["app_version"],
        "dateCreated": datetime.now().isoformat(),
        "license": "https://creativecommons.org/publicdomain/zero/1.0/",
        "creator": {
            "@type": "SoftwareApplication",
            "name": CONFIG["app_name"],
            "version": CONFIG["app_version"],
            "url": CONFIG["app_url"],
            "license": "https://www.gnu.org/licenses/agpl-3.0.html",
            "applicationCategory": "Scientific Research Tool",
            "operatingSystem": "Platform Independent",
            "softwareRequirements": "Python 3.13+, Marimo",
        },
        "provider": [
            {
                "@type": "Organization",
                "name": "LOTUS Initiative",
                "url": WIKIDATA_WIKI_URL + "Q104225190",
            },
            {
                "@type": "Organization",
                "name": "Wikidata",
                "url": WIKIDATA_URL,
            },
            {
                "@type": "Organization",
                "name": "IDSM (Integrated Database of Small Molecules)",
                "url": "https://idsm.elixir-czech.cz/",
            },
        ],
        "citation": [
            {
                "@type": "ScholarlyArticle",
                "name": "The LOTUS initiative for open knowledge management in natural products research",
                "identifier": "https://doi.org/10.7554/eLife.70780",
            }
        ],
        "distribution": [
            {
                "@type": "DataDownload",
                "encodingFormat": "text/csv",
                "contentUrl": "data:text/csv",
            },
            {
                "@type": "DataDownload",
                "encodingFormat": "application/json",
                "contentUrl": "data:application/json",
            },
            {
                "@type": "DataDownload",
                "encodingFormat": "text/turtle",
                "contentUrl": "data:text/turtle",
            },
        ],
        "numberOfRecords": len(df),
        "variablesMeasured": [
            "compound_name",
            "compound_smiles",
            "compound_inchikey",
            "compound_mass",
            "molecular_formula",
            "taxon_name",
            "reference_title",
            "reference_doi",
            "reference_date",
            "compound_qid",
            "taxon_qid",
            "reference_qid",
        ],
        "search_parameters": {
            "taxon": taxon_input,
            "taxon_qid": qid,
        },
        "sparql_endpoint": CONFIG["sparql_endpoint"],
        "chemical_search_service": {
            "name": "SACHEM",
            "provider": "IDSM",
            "endpoint": CONFIG["idsm_endpoint"],
            "capabilities": ["substructure_search", "similarity_search"],
        },
    }

    # Add filters if any are active
    if filters:
        metadata["search_parameters"]["filters"] = filters

    # Add provenance hashes for reproducibility and verification
    if query_hash or result_hash:
        metadata["provenance"] = {}
        if query_hash:
            metadata["provenance"]["query_hash"] = {
                "algorithm": "SHA-256",
                "value": query_hash,
                "description": "Hash of search parameters (what was asked) - allows query reproduction",
            }
        if result_hash:
            metadata["provenance"]["result_hash"] = {
                "algorithm": "SHA-256",
                "value": result_hash,
                "description": "Hash of result compound identifiers (what was found) - content-addressable dataset identifier",
            }
            # Dataset URI is based ONLY on result hash (content-addressable)
            metadata["provenance"]["dataset_uri"] = f"urn:hash:sha256:{result_hash}"
            metadata["provenance"]["dataset_uri_note"] = (
                "This URI identifies the dataset by its content (results), "
                "independent of how it was obtained (query). "
                "The query_hash above documents how this dataset was generated."
            )

    return metadata


def create_citation_text(taxon_input: str) -> str:
    """
    Generate citation text for the exported data.

    Args:
        taxon_input: Taxon name or identifier

    Returns:
        Formatted citation text in markdown
    """
    current_date = datetime.now().strftime("%B %d, %Y")
    return f"""
## 📖 How to Cite This Data

### Dataset Citation
LOTUS Initiative via Wikidata. ({datetime.now().year}). *Data for {taxon_input}*.  
Retrieved from LOTUS Wikidata Explorer on {current_date}.  
License: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)

### LOTUS Initiative Publication
Rutz A, Sorokina M, Galgonek J, et al. (2022). The LOTUS initiative for open knowledge 
management in natural products research. *eLife* **11**:e70780.  
DOI: [10.7554/eLife.70780](https://doi.org/10.7554/eLife.70780)

### This Tool
{CONFIG["app_name"]} v{CONFIG["app_version"]}  
[Source Code]({CONFIG["app_url"]}) (AGPL-3.0)

### Data Sources
- **LOTUS Initiative**: [Q104225190](https://www.wikidata.org/wiki/Q104225190) - CC0 1.0
- **Wikidata**: [www.wikidata.org](https://www.wikidata.org/) - CC0 1.0
"""
