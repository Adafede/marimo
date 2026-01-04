"""
Application configuration and settings.

All magic numbers, thresholds, and configurable parameters are centralized here
for easy maintenance and tuning.
"""

from typing import Dict, Any, List, Tuple, Callable
import polars as pl

# ====================================================================
# APPLICATION CONFIGURATION
# All magic numbers centralized here for easy maintenance.
# ====================================================================

CONFIG: Dict[str, Any] = {
    # Application Metadata
    "app_version": "0.0.1",
    "app_name": "LOTUS Wikidata Explorer",
    "app_url": "https://github.com/Adafede/marimo/blob/main/apps/lotus_wikidata_explorer.py",
    # External Services
    "cdk_base": "https://www.simolecule.com/cdkdepict/depict/cot/svg",  # CDK Depict for structure images
    "sparql_endpoint": "https://qlever.dev/api/wikidata",  # QLever: Optimized Wikidata SPARQL endpoint (faster than official)
    # Alternative endpoint (legacy, slower but more stable):
    # "sparql_endpoint": "https://query-legacy-full.wikidata.org/sparql",
    "idsm_endpoint": "https://idsm.elixir-czech.cz/sparql/endpoint/",  # SACHEM chemical search service
    # Network & Performance Tuning
    "max_retries": 3,  # Max retry attempts for failed SPARQL requests (prevents transient failures)
    "retry_backoff": 2,  # Exponential backoff multiplier (wait time = backoff^attempt seconds)
    "query_timeout": 300,  # SPARQL query timeout in seconds (5 minutes - some queries are complex)
    "table_row_limit": 10000,  # Max rows to display in browser table (prevents UI slowdown with large datasets)
    "lazy_generation_threshold": 5000,  # Defer download generation for datasets > this size (improves UX)
    "download_embed_threshold_bytes": 8_000_000,  # Auto-compress downloads > 8MB (reduces bandwidth)
    # UI Styling & Display
    "color_hyperlink": "#3377c4",  # Hyperlink color (WCAG AA compliant blue)
    "color_wikidata_blue": "#006699",
    "color_wikidata_green": "#339966",
    "color_wikidata_red": "#990000",
    "page_size_default": 10,  # Rows per page in display table (balances usability and performance)
    "page_size_export": 25,  # Rows per page in export preview table (larger for review)
    # Filter Defaults (Scientific domain knowledge)
    "year_range_start": 1700,  # Minimum valid publication year (pre-Linnaean taxonomy excluded)
    "year_default_start": 1900,  # Default year filter start (modern natural products research era)
    "mass_default_min": 0,  # Default minimum mass in Daltons (no restriction)
    "mass_default_max": 2000,  # Default maximum mass in Daltons (typical natural product range)
    "mass_ui_max": 10000,  # Maximum allowed mass in UI (prevents unrealistic values)
    # Element Count Limits (for molecular formula filter UI)
    # Based on natural product chemistry - typical ranges for drug-like molecules
    "element_c_max": 100,  # Max carbon atoms (most natural products < 100)
    "element_h_max": 200,  # Max hydrogen atoms (roughly 2× carbon)
    "element_n_max": 50,  # Max nitrogen atoms (alkaloids rarely exceed this)
    "element_o_max": 50,  # Max oxygen atoms (polyketides/carbohydrates)
    "element_p_max": 20,  # Max phosphorus atoms (nucleotides)
    "element_s_max": 20,  # Max sulfur atoms (sulfated metabolites)
    # Chemical Search
    "default_similarity_threshold": 0.8,  # Default Tanimoto coefficient (0.8 = good balance of recall/precision)
}

# ====================================================================
# URL CONSTANTS
# ====================================================================

SCHOLIA_URL = "https://scholia.toolforge.org/"
WIKIDATA_URL = "https://www.wikidata.org/"
WIKIDATA_HTTP_URL = WIKIDATA_URL.replace("https://", "http://")
WIKIDATA_ENTITY_URL = WIKIDATA_HTTP_URL + "entity/"
WIKIDATA_WIKI_URL = WIKIDATA_URL + "wiki/"

# ====================================================================
# ELEMENT CONFIGURATION
# ====================================================================

# Element definitions for formula filters (avoid hardcoding element lists)
ELEMENT_CONFIGS: List[Tuple[str, str, str]] = [
    ("C", "carbon", "element_c_max"),
    ("H", "hydrogen", "element_h_max"),
    ("N", "nitrogen", "element_n_max"),
    ("O", "oxygen", "element_o_max"),
    ("P", "phosphorus", "element_p_max"),
    ("S", "sulfur", "element_s_max"),
]

HALOGEN_CONFIGS: List[Tuple[str, str]] = [
    ("F", "fluorine"),
    ("Cl", "chlorine"),
    ("Br", "bromine"),
    ("I", "iodine"),
]

# ====================================================================
# EXPORT CONFIGURATION
# ====================================================================

# Export format configurations
EXPORT_FORMATS: Dict[str, Dict[str, Any]] = {
    "csv": {
        "extension": "csv",
        "mimetype": "text/csv",
        "label": "📥 CSV",
        "icon": "📄",
        "generator": lambda df: df.write_csv(),
    },
    "json": {
        "extension": "json",
        "mimetype": "application/json",
        "label": "📥 JSON",
        "icon": "📖",
        "generator": lambda df: df.write_json(),
    },
    "ttl": {
        "extension": "ttl",
        "mimetype": "text/turtle",
        "label": "📥 RDF/Turtle",
        "icon": "🐢",
        "generator": None,  # Needs special handling (extra params)
    },
}
