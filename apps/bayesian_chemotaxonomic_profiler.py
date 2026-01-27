# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "altair==6.0.0",
#     "cmcrameri==1.9",
#     "marimo",
#     "numpy==2.4.1",
#     "polars==1.37.1",
#     "pyarrow==23.0.0",
#     "scipy==1.17.0",
#     "simple-parsing==0.1.7",
#     # "vegafusion==2.0.3",
#     # "vl-convert-python==1.9.0.post1",
# ]
# ///
"""
Bayesian Chemotaxonomic Scaffold Discovery

Discover scaffold-taxon associations while accounting for sampling bias.

===========================================================================
FOR NON-STATISTICIANS: What This Tool Does
===========================================================================

QUESTION: "Is this chemical scaffold PRACTICALLY enriched in this taxon?"

CHALLENGE: Our data is extremely sparse:
- Only a very small number of known taxa have any chemistry reported
- Most studied taxa have only 3-5 compounds reported
- Most compounds are known from only 1-2 species

NAIVE APPROACH (wrong): Count how many taxa have the scaffold vs don't.
→ Problem: This treats "unstudied" as "doesn't have it", which is unfair.

OUR APPROACH: Bayesian inference with ROPE-based decision-making
1. Only count INVESTIGATED taxa as evidence
2. Weight observations by diversity (4 compounds from 4 species > 4 from 1)
3. Use 89% credible intervals following Kruschke (2015)
4. Use ROPE (Region of Practical Equivalence) for decisions

===========================================================================
KEY CONCEPTS
===========================================================================

ROPE (Region of Practical Equivalence)
--------------------------------------
Instead of asking "is the effect different from zero?" (point-null), ROPE
asks "is the effect PRACTICALLY different from baseline?"

ROPE = [-ε, +ε] on log₂ fold-change scale (default ε = 0.1)
This means fold-changes between 0.93× and 1.07× are "no practical effect"

DECISIONS:
- enriched:   CI entirely above +ε → practically enriched
- depleted:   CI entirely below -ε → practically depleted
- equivalent: CI entirely within ROPE → no practical difference
- undecided:  CI overlaps ROPE boundaries → need more data

Credible Interval (CI)
----------------------
We use 89% CI following Kruschke (2015):
- 89% is a prime number, reminding us the choice is arbitrary
- 95% is a frequentist convention with no Bayesian justification
- We compute equal-tailed intervals, which equal HDI for symmetric distributions

ESS (Effective Sample Size)
---------------------------
ESS = (α_post + β_post) - (α_prior + β_prior)
Measures how much the data contributed beyond the prior.
ESS < 3 indicates insufficient data for reliable inference.

===========================================================================

Model
-----
Prior:     Beta(θ₀λ, (1-θ₀)λ) centered on scaffold's observed frequency
Posterior: Beta(α₀ + a_eff, β₀ + c_eff)
Decision:  Based on CI position relative to ROPE

References
----------
- John K. Kruschke (2015) Doing Bayesian Data Analysis (Second Edition). Academic Press. https://doi.org/10.1016/B978-0-12-405888-0.00001-5
- Gelman et al. (2008) A weakly informative default prior distribution for logistic and other regression models. The Annals of Applied Statistics. 2(4):1360-1383. https://doi.org/10.1214/08-AOAS191
- Rutz et al. (2022) The LOTUS initiative for open knowledge management in natural products research. eLife 11:e70780. https://doi.org/10.7554/eLife.70780


Copyright (C) 2026 Adriano Rutz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium", app_title="Bayesian Chemotaxonomic Markers")

with app.setup:
    import json
    import logging
    import sys
    import time
    import urllib.request
    import urllib.parse
    from dataclasses import dataclass, field
    from pathlib import Path
    from typing import Any, Iterable, Final, Sequence, TypedDict

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    from scipy.special import betainc, betaincinv
    from urllib.parse import quote

    from modules.chem.cdk.depict.svg_from_smiles import svg_from_smiles
    from modules.knowledge.wikidata.taxon.ranks import get_rank_label, get_rank_order
    from modules.net.sparql.execute_with_retry import execute_with_retry
    from modules.net.sparql.parse_response import parse_sparql_response
    from modules.stats.bayesian.beta import (
        credible_interval,
        posterior_probability_above,
    )
    from modules.stats.bayesian.enrichment import (
        fold_change_credible_interval,
        rope_decision,
    )

    alt.data_transformers.enable("vegafusion")

    # Patch urllib for Pyodide/WASM (browser) compatibility
    IS_PYODIDE = "pyodide" in sys.modules
    if IS_PYODIDE:
        import pyodide_http

        pyodide_http.patch_all()

    class StatisticalConstantsDict(TypedDict):
        """Core statistical constants used throughout the analysis."""

        # Laplace smoothing for ratio computation
        continuity_correction: float
        # Minimum baseline rate to prevent numerical issues
        min_theta_0: float
        # Credible interval probability mass (89% following Kruschke 2015)
        ci_prob: float
        # ROPE half-width on log2 fold-change scale
        rope_half_width: float
        # Minimum effective sample size for reliable inference
        min_ess: int

    class FilteringThresholdsDict(TypedDict):
        """Minimal filtering thresholds for feature and group inclusion."""

        # Minimum compounds required for a scaffold to be analyzed
        min_frequency_scaffold: int
        # Minimum compounds required for a taxon to be analyzed
        min_frequency_taxa: int
        # Whether to include full compounds as scaffolds
        include_compounds_as_scaffolds: bool

    class BayesianPriorsDict(TypedDict):
        """Bayesian prior parameters for enrichment analysis."""

        # Prior strength λ (pseudo-observation count)
        prior_strength: float
        # Enable hierarchical prior flow from parent to child taxa
        hierarchical_prior_flow: bool
        # Weight for parent posterior in hierarchical prior (0-1)
        hierarchical_weight: float

    class DataPathsDict(TypedDict):
        """File paths for input data."""

        path_can_smi: str
        path_frags_cdk: str
        path_frags_ert: str
        path_frags_sru: str
        path_items_cdk: str
        path_items_ert: str
        path_items_sru: str

    class EnrichmentConfigDict(TypedDict):
        """Complete configuration for chemotaxonomic enrichment analysis."""

        # SPARQL endpoint
        qlever_endpoint: str
        # Data paths
        data_paths: DataPathsDict
        # Statistical constants
        stats: StatisticalConstantsDict
        # Filtering thresholds
        filtering: FilteringThresholdsDict
        # Bayesian priors
        priors: BayesianPriorsDict

    # ========================================================================
    # STATISTICAL CONSTANTS
    # ========================================================================
    # All "magic numbers" are defined here for transparency and easy adjustment.

    # CONTINUITY_CORRECTION = 0.5 (Laplace smoothing)
    # ------------------------------------------------
    # When computing ratios like sensitivity = a/(a+c), we add 0.5 to avoid
    # division by zero and extreme values from small samples. This is
    # "Laplace smoothing" or "add-half" correction.
    # Value of 0.5 is conventional (Agresti & Coull, 1998)
    # https://doi.org/10.1080/00031305.1998.10480550

    # MIN_THETA_0 = 1e-6 (Baseline floor)
    # -----------------------------------
    # The baseline rate θ₀ is the scaffold's observed frequency. For extremely
    # rare scaffolds, this could approach zero, causing numerical issues.
    # Floor of 1e-6 (0.0001%) prevents division by zero.

    # CI_PROB = 0.89 (Credible Interval probability mass)
    # --------------------------------------------------
    # Following Kruschke (2015):
    # - 89% is a prime number, reminding us the choice is arbitrary
    # - 95% is a frequentist convention with no Bayesian justification
    # We compute Equal-Tailed Intervals (ETI), which equal HDI for symmetric
    # distributions and are a close approximation for moderately skewed ones.

    # ROPE_HALF_WIDTH = 0.1 (Region of Practical Equivalence)
    # -------------------------------------------------------
    # ROPE defines "practically equivalent to baseline" as [θ₀ - ε, θ₀ + ε].
    # Default ε = 0.1 means ±10% on log2 scale is considered "no meaningful effect".
    #
    # Example: If baseline θ₀ = 0.05 (5%), ROPE is [0.04, 0.06] on log2 scale.
    # A posterior entirely above ROPE → practically enriched.
    # A posterior overlapping ROPE → undecided.
    # A posterior entirely below ROPE → practically depleted.
    #
    # The value 0.1 (≈ ±0.14 log2 fold-change) follows Cohen's "small effect"
    # convention adapted to proportions. Adjust based on domain knowledge.

    # MIN_ESS = 3 (Minimum Effective Sample Size for "reliable" inference)
    # --------------------------------------------------------------------
    # The "rule of three": with fewer than 3 observations, we cannot reliably
    # distinguish signal from noise. This is NOT arbitrary:
    # - With n=1, any binary outcome has 100% frequency
    # - With n=2, we can only observe 0%, 50%, or 100%
    # - With n=3, we start to see meaningful variation
    # This threshold flags results as "needs more data" rather than filtering.

    STATISTICAL_CONSTANTS: Final[StatisticalConstantsDict] = {
        "continuity_correction": 0.5,
        "min_theta_0": 1e-6,
        "ci_prob": 0.89,
        "rope_half_width": 0.1,
        "min_ess": 3,
    }

    # ====================================================================
    # FILTERING THRESHOLDS
    # ====================================================================
    # We use MINIMAL filtering to let the Bayesian model handle edge cases.
    # The model naturally assigns low confidence (wide CI) to poorly-sampled
    # cases, so aggressive pre-filtering would just lose information.

    # min_frequency_scaffold = 2: Require scaffold in ≥2 compounds
    # Why 2? A scaffold appearing in only 1 compound gives no statistical
    # power - we can't distinguish "enriched" from "noise" with n=1.

    # min_frequency_taxa = 2: Require taxon to have ≥2 compounds reported
    # Same reasoning: with only 1 compound, any scaffold is either 100% or 0%
    # present, which is meaningless for enrichment analysis.

    FILTERING_THRESHOLDS: Final[FilteringThresholdsDict] = {
        "min_frequency_scaffold": 2,
        "min_frequency_taxa": 2,
        "include_compounds_as_scaffolds": True,
    }

    # ====================================================================
    # BAYESIAN PRIOR PARAMETERS
    # ====================================================================
    #
    # PRIOR STRENGTH (λ) = 1.0: How much weight does our prior belief get?
    # --------------------------------------------------------------------
    # Think of this as "pseudo-observations" from prior knowledge.
    # λ=1 means the prior contributes roughly 1 pseudo-observation.
    #
    # Why 1 and not 5, 10, or 20?
    # - λ=1 is a "weakly informative" prior (Gelman et al., 2008)
    # - With λ=1, a single real observation already outweighs the prior
    # - This is appropriate for our sparse data where most taxa have
    #   only 1-5 compounds reported
    # - Higher values (5, 10) would require more data to "overcome" the
    #   prior, making small but real signals harder to detect
    #
    # We initially tested λ=5 (a common default) but found it:
    # - Required 5+ observations to show significant enrichment
    # - Suppressed real signals in genera with only 2-3 compounds
    # - For our SPARSE data, this was too conservative
    #
    # We also tested "adaptive priors" that scaled λ with scaffold frequency
    # (e.g., λ = base × log₂(total+1), capped at 20) but found:
    # - Ubiquitous scaffolds got λ≈15-20, overwhelming small sample signals
    # - The cap of 20 was arbitrary with no principled justification
    # - The true Bayesian P(θ>θ₀) formula already handles ubiquitous
    #   scaffolds correctly via the high θ₀ baseline
    #
    # CONCLUSION: Use constant λ=1, let θ₀ handle ubiquity naturally.

    # HIERARCHICAL PRIOR FLOW: Should child taxa inherit parent information?
    # --------------------------------------------------------------------
    # When True, priors at lower ranks (Genus, Species) are influenced by
    # posteriors at higher ranks (Family, Order).
    #
    # INTERPRETATION: All comparisons are against the GLOBAL baseline θ₀.
    # - If a scaffold is enriched in Gentianaceae vs the dataset,
    #   it's MORE likely to be enriched in Gentiana vs the dataset.
    # - NOT testing "enriched in Gentiana vs other Gentianaceae genera"
    #
    # The prior center becomes:
    #   prior_center = w × parent_posterior + (1-w) × θ₀
    #
    # This provides gentle guidance: "Your parent family is enriched, so you
    # (child genus) are slightly more likely to be enriched too, but your
    # own data will dominate."

    # HIERARCHICAL WEIGHT (w) = 0.1: How much does parent influence child?
    # -------------------------------------------------------------------
    # The prior center for a child taxon is:
    #   prior_center = w × parent_posterior + (1-w) × global_baseline
    #
    # With w=0.1:
    # - 90% of the prior comes from global baseline (the scaffold's frequency)
    # - 10% comes from the parent taxon's posterior
    #
    # Why 0.1 (10%) and not 0.5 (50%)?
    # - We want the CHILD'S OWN DATA to dominate, not the parent's
    # - Parent information is a gentle nudge, not a strong constraint
    # - Higher values would make it hard for a genus to show different
    #   enrichment than its family, even with strong evidence
    # - 0.1 is a common choice for "weak shrinkage" in hierarchical models

    BAYESIAN_PRIORS: Final[BayesianPriorsDict] = {
        "prior_strength": 1.0,
        "hierarchical_prior_flow": True,
        "hierarchical_weight": 0.1,
    }

    # ====================================================================
    # DEFAULT DATA PATHS
    # ====================================================================

    DEFAULT_DATA_PATHS: Final[DataPathsDict] = {
        "path_can_smi": "apps/public/mortar/lotus_canonical.smi.gz",
        "path_frags_cdk": "apps/public/mortar/Fragments_Scaffold_Generator.csv.gz",
        "path_frags_ert": "apps/public/mortar/Fragments_Ertl_algorithm.csv.gz",
        "path_frags_sru": "apps/public/mortar/Fragments_Sugar_Removal_Utility.csv.gz",
        "path_items_cdk": "apps/public/mortar/Items_Scaffold_Generator.csv.gz",
        "path_items_ert": "apps/public/mortar/Items_Ertl_algorithm.csv.gz",
        "path_items_sru": "apps/public/mortar/Items_Sugar_Removal_Utility.csv.gz",
    }

    # ====================================================================
    # COMPLETE DEFAULT CONFIGURATION
    # ====================================================================

    DEFAULT_CONFIG: Final[EnrichmentConfigDict] = {
        "qlever_endpoint": "https://qlever.dev/api/wikidata",
        "data_paths": DEFAULT_DATA_PATHS,
        "stats": STATISTICAL_CONSTANTS,
        "filtering": FILTERING_THRESHOLDS,
        "priors": BAYESIAN_PRIORS,
    }
    STANDARD_RANKS = [
        "Kingdom",
        "Phylum",
        "Class",
        "Order",
        "Family",
        "Genus",
        "Species",
    ]
    FOCAL_RANKS = ["Kingdom", "Order", "Family", "Genus"]

    MIN_OBS = 3
    MIN_PROB = 0.89
    MIN_LOG2FC = 1.5

    TOP_N_TAXA = 10
    TOP_N_MARKERS_PER_TAXON = 10


@app.function
def smiles_to_img_urls(smiles_list):
    q = quote  # local binding = faster
    base = "https://www.simolecule.com/cdkdepict/depict/cow/svg"

    return [f"{base}?smi={q(s)}" for s in smiles_list]


@app.function
def smiles_to_thumbs(smiles_list, size: int = 120):
    urls = smiles_to_img_urls(smiles_list)
    return [mo.image(src=url, width=size, height=size) for url in urls]


@app.function
def validate_columns(
    df: pl.DataFrame,
    required: Iterable[str],
    name: str,
) -> pl.DataFrame:
    """Validate dataframe has required columns with helpful error messages."""
    if df is None or df.is_empty():
        logging.warning(f"{name}: Empty dataframe")
        return df
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"{name}: Missing columns {sorted(missing)}\n"
            f"  Available: {sorted(df.columns)}\n"
            f"  Required: {sorted(required)}",
        )
    logging.debug(f"{name}: ✓ {df.height:,} rows")
    return df


@app.function
def read_table(
    path: Path,
    separator: str = ",",
    expected: Iterable[str] | None = None,
    name: str = "table",
) -> pl.DataFrame:
    """Read CSV and optionally validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    df = pl.read_csv(path, separator=separator)
    return validate_columns(df, expected, name) if expected else df


@app.function
def select_first_existing(cfg: Final[TypedDict], attrs: Sequence[str]) -> Path | None:
    """Return the first path from cfg attributes that exists on disk."""
    for attr in attrs:
        p = Path(cfg[attr])
        if p.exists():
            return p
    return None


@app.function
def load_fragments(path: Path, min_freq: int) -> pl.DataFrame:
    """Load fragment SMILES, filtering by minimum molecule frequency."""
    return read_table(
        path,
        expected={"SMILES", "MoleculeFrequency"},
        name="fragments",
    ).filter(pl.col("MoleculeFrequency") >= min_freq)


@app.function
def load_compound_fragment_mapping(path: Path) -> pl.DataFrame:
    """Parse compound-to-fragment mapping file."""
    return (
        pl.read_csv(
            path,
            new_columns=["raw"],
            truncate_ragged_lines=True,
            separator="\n",
        )
        .with_columns(
            [
                pl.col("raw").str.extract(r"^([^,]+)", 1).alias("compound_name"),
                pl.col("raw").str.replace(r"^[^,]+,", "").alias("fragments_str"),
            ],
        )
        .select(["compound_name", "fragments_str"])
    )


@app.function
def build_compound_scaffold_table(
    compounds: pl.DataFrame,
    mapping: pl.DataFrame,
    fragments: pl.DataFrame,
) -> pl.DataFrame:
    """Build compound-scaffold relationship table from fragment mapping."""
    if (
        mapping is None
        or mapping.is_empty()
        or compounds.is_empty()
        or fragments.is_empty()
    ):
        return pl.DataFrame({"compound_smiles": [], "scaffold": []})
    unique = compounds.select("compound_smiles").unique()
    exploded = (
        pl.concat([unique, mapping], how="horizontal")
        .with_columns(pl.col("fragments_str").str.split(","))
        .explode("fragments_str")
        .rename({"fragments_str": "scaffold"})
    )
    return (
        exploded.filter(pl.col("scaffold").is_in(fragments["SMILES"].to_list()))
        .select(["compound_smiles", "scaffold"])
        .unique()
    )


@app.function
def build_hierarchy_lineage(
    edges: pl.DataFrame,
    child: str,
    parent: str,
    focus: pl.Series | None = None,
) -> pl.DataFrame:
    """Build ancestor lineage from parent-child edges via BFS traversal."""
    validate_columns(edges, {child, parent}, "hierarchy")
    e = edges.select([child, parent]).drop_nulls().unique()
    if e.is_empty():
        return pl.DataFrame({child: [], "ancestor": [], "distance": []})

    nodes = (
        pl.DataFrame({child: focus.drop_nulls().unique()})
        if focus is not None
        else pl.DataFrame({child: e.get_column(child).unique()})
    )
    current = (
        nodes.join(e, on=child, how="inner")
        .rename({parent: "ancestor"})
        .with_columns(pl.lit(1).cast(pl.Int32).alias("distance"))
        .unique()
    )

    rows, visited = (
        [],
        pl.DataFrame(
            {
                child: pl.Series([], dtype=e.schema[child]),
                "ancestor": pl.Series([], dtype=e.schema[parent]),
            },
        ),
    )
    for _ in range(50):  # Max depth
        if current.is_empty():
            break
        rows.append(current)
        visited = pl.concat([visited, current.select(child, "ancestor")]).unique()
        current = (
            current.join(e, left_on="ancestor", right_on=child, how="inner")
            .select(
                [
                    pl.col(child),
                    pl.col(parent).alias("ancestor"),
                    (pl.col("distance") + 1).cast(pl.Int32).alias("distance"),
                ],
            )
            .join(visited, on=[child, "ancestor"], how="anti")
            .unique()
        )

    lineage = (
        pl.concat(rows).unique()
        if rows
        else pl.DataFrame({child: [], "ancestor": [], "distance": []})
    )
    all_nodes = nodes.get_column(child).unique()
    self_loops = pl.DataFrame(
        {
            child: all_nodes,
            "ancestor": all_nodes,
            "distance": pl.Series([0] * len(all_nodes), dtype=pl.Int32),
        },
    )
    return pl.concat([lineage, self_loops]).unique().sort([child, "distance"])


@app.function
def build_taxon_lineage(
    taxon_parent: pl.DataFrame,
    taxon_rank: pl.DataFrame | None = None,
    focus: pl.Series | None = None,
) -> pl.DataFrame:
    """Build taxon ancestor lineage with optional rank annotation."""
    lineage = build_hierarchy_lineage(taxon_parent, "taxon", "taxon_parent", focus)
    lineage = lineage.rename(
        {"ancestor": "taxon_ancestor", "distance": "taxon_distance"},
    )
    if taxon_rank is not None:
        lineage = lineage.join(
            taxon_rank.rename({"taxon": "taxon_ancestor"}),
            on="taxon_ancestor",
            how="left",
        )
    return lineage


@app.function
def build_compound_lineage(compound_scaffold: pl.DataFrame) -> pl.DataFrame:
    """Map compounds to their scaffolds (flat, distance=0)."""
    validate_columns(compound_scaffold, {"compound", "scaffold"}, "compound_scaffold")
    return compound_scaffold.select(
        [
            pl.col("compound"),
            pl.col("scaffold").alias("compound_ancestor"),
            pl.lit(0).cast(pl.Int32).alias("compound_distance"),
        ],
    ).unique()


@app.function
def run_hierarchical_analysis(
    compound_taxon: pl.DataFrame,
    compound_lineage: pl.DataFrame,
    taxon_lineage: pl.DataFrame,
    taxon_name: pl.DataFrame,
    taxon_rank: pl.DataFrame | None = None,
    cfg: Final[TypedDict] = None,
) -> pl.DataFrame:
    """
    Run Bayesian enrichment analysis across taxonomic ranks.

    For each scaffold-taxon pair:
    1. Count compounds with/without scaffold in investigated taxa only
    2. Apply diversity weighting: a_eff = √(compounds × source_taxa)
    3. Compute P(enriched | data) using Beta-Binomial model

    Optional rank-to-rank priors blend parent posterior with global baseline.
    """
    start = time.time()
    if cfg is None:
        cfg = DEFAULT_CONFIG

    logging.info("=" * 55)
    logging.info("CHEMOTAXONOMIC SCAFFOLD ANALYSIS")
    logging.info("=" * 55)

    # Build base evidence: compound × scaffold × original taxon
    base_evidence = (
        compound_taxon.join(compound_lineage, on="compound", how="inner")
        .select(["compound", "taxon", "compound_ancestor"])
        .unique()
    )

    if base_evidence.is_empty():
        return pl.DataFrame()

    # Minimal scaffold filtering - only require appearing in ≥2 compounds
    # The Bayesian model handles ubiquitous scaffolds through θ₀ (baseline)
    scaffold_freq = base_evidence.group_by("compound_ancestor").agg(
        [
            pl.col("compound").n_unique().alias("n_compounds"),
        ],
    )

    keep_scaffolds = scaffold_freq.filter(
        pl.col("n_compounds") >= cfg["filtering"]["min_frequency_scaffold"],
    ).select("compound_ancestor")

    logging.info(
        f"Scaffolds: {keep_scaffolds.height} (≥{cfg["filtering"]["min_frequency_scaffold"]} compounds)",
    )

    base_evidence = base_evidence.join(
        keep_scaffolds,
        on="compound_ancestor",
        how="inner",
    )

    if base_evidence.is_empty():
        return pl.DataFrame()

    N = int(base_evidence.get_column("compound").n_unique())
    logging.info(f"Universe: {N} compounds, {keep_scaffolds.height} scaffolds")

    # Get valid ranks ordered coarse→fine
    valid_ranks_df = (
        taxon_lineage.filter(pl.col("taxon_rank").is_not_null())
        .group_by("taxon_rank")
        .agg(pl.col("taxon_ancestor").n_unique().alias("n_taxa"))
        .filter(pl.col("n_taxa") >= 3)
    )

    if valid_ranks_df.is_empty():
        return pl.DataFrame()

    valid_ranks = valid_ranks_df.get_column("taxon_rank").to_list()
    # Keep only ranks that exist in our internal rank-order dictionary
    valid_ranks = [r for r in valid_ranks if get_rank_order(r) < 999]
    if not valid_ranks:
        logging.warning(
            "No valid ranks found within the internal rank dictionary - aborting analysis"
        )
        return pl.DataFrame()

    valid_ranks = sorted(valid_ranks, key=lambda r: get_rank_order(r))

    logging.info(f"Ranks (coarse→fine): {valid_ranks}")

    # Build parent-child mappings between ranks for hierarchical priors
    taxon_rank_map = (
        taxon_lineage.select(["taxon_ancestor", "taxon_rank"])
        .unique()
        .filter(pl.col("taxon_rank").is_in(valid_ranks))
    )

    parent_child_maps = {}
    for i in range(1, len(valid_ranks)):
        child_rank = valid_ranks[i]
        parent_rank = valid_ranks[i - 1]

        child_taxa = (
            taxon_rank_map.filter(pl.col("taxon_rank") == child_rank)
            .select("taxon_ancestor")
            .rename({"taxon_ancestor": "child_taxon"})
        )

        parent_map = (
            taxon_lineage.filter(pl.col("taxon_rank") == parent_rank)
            .select(["taxon", "taxon_ancestor"])
            .rename({"taxon_ancestor": "parent_taxon"})
            .join(
                child_taxa.rename({"child_taxon": "taxon"}),
                on="taxon",
                how="inner",
            )
            .select(["taxon", "parent_taxon"])
            .rename({"taxon": "child_taxon"})
            .unique()
        )

        parent_child_maps[child_rank] = parent_map

    # Process ranks with rank-to-rank prior propagation
    rank_results = []
    posteriors_by_rank = {}

    for rank_idx, rank in enumerate(valid_ranks):
        logging.info(f"  Processing {rank}")

        # Map each taxon to its ancestor at this rank
        rank_map = (
            taxon_lineage.filter(pl.col("taxon_rank") == rank)
            .select(["taxon", "taxon_ancestor"])
            .unique()
        )

        logging.info(f"    {rank_map.height} taxon-ancestor mappings")

        # Propagate compounds up to this rank level
        rank_evidence = (
            base_evidence.join(rank_map, on="taxon", how="inner")
            .select(
                [
                    "compound",
                    "taxon_ancestor",
                    "compound_ancestor",
                    pl.col("taxon").alias("source_taxon"),
                ],
            )
            .unique()
        )

        logging.info(f"{rank_evidence.get_column('compound').n_unique()} compounds")

        if rank_evidence.is_empty():
            logging.warning(f"    No evidence for rank {rank} - skipping")
            continue

        # Scaffold totals before filtering
        scaffold_totals_full = rank_evidence.group_by("compound_ancestor").agg(
            pl.col("compound").n_unique().alias("scaffold_total"),
        )

        # Filter taxa by minimum frequency
        taxon_freq = rank_evidence.group_by("taxon_ancestor").agg(
            pl.col("compound").n_unique().alias("taxon_obs"),
        )
        keep_taxa = taxon_freq.filter(
            pl.col("taxon_obs") >= cfg["filtering"]["min_frequency_taxa"],
        ).select("taxon_ancestor")
        rank_evidence_filtered = rank_evidence.join(
            keep_taxa,
            on="taxon_ancestor",
            how="inner",
        )

        if rank_evidence_filtered.is_empty():
            continue

        # Diversity weighting: a_eff = √(compounds × source_taxa)
        a_counts = (
            rank_evidence_filtered.group_by(["compound_ancestor", "taxon_ancestor"])
            .agg(
                [
                    pl.col("compound").n_unique().alias("a_raw"),
                    pl.col("source_taxon").n_unique().alias("n_source_taxa"),
                ],
            )
            .with_columns(
                (
                    pl.col("a_raw").cast(pl.Float32).sqrt()
                    * pl.col("n_source_taxa").cast(pl.Float32).sqrt()
                )
                .round(0)
                .cast(pl.Int32)
                .clip(lower_bound=1)
                .alias("a"),
            )
        )

        taxon_totals = rank_evidence_filtered.group_by("taxon_ancestor").agg(
            pl.col("compound").n_unique().alias("taxon_total"),
        )

        rank_data = (
            a_counts.join(scaffold_totals_full, on="compound_ancestor", how="left")
            .join(taxon_totals, on="taxon_ancestor", how="left")
            .with_columns(
                [
                    (pl.col("scaffold_total") - pl.col("a_raw"))
                    .clip(lower_bound=0)
                    .alias("b"),
                    (pl.col("taxon_total") - pl.col("a_raw"))
                    .clip(lower_bound=0)
                    .alias("c"),
                    (
                        (pl.col("taxon_total") - pl.col("a_raw")).cast(pl.Float32)
                        * (
                            pl.col("a").cast(pl.Float32)
                            / (pl.col("a_raw").cast(pl.Float32) + 1e-10)
                        )
                    )
                    .round(0)
                    .cast(pl.Int32)
                    .clip(lower_bound=0)
                    .alias("c_eff"),
                    (
                        pl.lit(N)
                        - pl.col("scaffold_total")
                        - pl.col("taxon_total")
                        + pl.col("a_raw")
                    )
                    .clip(lower_bound=0)
                    .alias("d"),
                    pl.lit(N).alias("N"),
                    pl.lit(rank).alias("taxon_rank"),
                ],
            )
        )
        CONTINUITY_CORRECTION = cfg["stats"]["continuity_correction"]
        # Classical statistics
        rank_data = rank_data.with_columns(
            [
                (pl.col("a") / (pl.col("a") + pl.col("c") + CONTINUITY_CORRECTION))
                .clip(0, 1)
                .alias("sensitivity"),
                (pl.col("d") / (pl.col("b") + pl.col("d") + CONTINUITY_CORRECTION))
                .clip(0, 1)
                .alias("specificity"),
                (pl.col("a") / (pl.col("a") + pl.col("b") + CONTINUITY_CORRECTION))
                .clip(0, 1)
                .alias("precision"),
                (pl.col("b") / (pl.col("b") + pl.col("d") + CONTINUITY_CORRECTION))
                .clip(0, 1)
                .alias("fpr"),
            ],
        ).with_columns(
            [
                (pl.col("sensitivity") / (pl.col("fpr") + CONTINUITY_CORRECTION)).alias(
                    "likelihood_ratio",
                ),
            ],
        )

        # ================================================================
        # BASELINE θ₀: First Principles
        # ================================================================
        #
        # THE FUNDAMENTAL QUESTION:
        # We want to know P(θ > θ₀ | data) - the probability that the true
        # rate of this scaffold in this taxon exceeds some baseline θ₀.
        # But what should θ₀ BE?
        #
        # OPTION 1: θ₀ = 0.5 (Jeffreys uninformative prior)
        # -------------------------------------------------
        # From first principles, if we have NO prior belief about scaffold
        # frequency, the maximally uninformative baseline is θ₀ = 0.5.
        # This is the Jeffreys prior for a Bernoulli rate parameter.
        #
        # Problem: With θ₀ = 0.5, "enrichment" means "rate > 50%".
        # - A scaffold in 2/3 compounds (67%) → P(θ > 0.5) ≈ 0.85 ✓
        # - A scaffold in 2/100 compounds (2%) → P(θ > 0.5) ≈ 0.0 ✗
        #
        # But the second case IS enriched if this scaffold normally appears
        # in only 0.1% of compounds! The uninformative prior loses the
        # concept of "enriched RELATIVE TO what's expected."
        #
        # OPTION 2: θ₀ = observed frequency in dataset
        # ---------------------------------------------
        # This is what we use. θ₀ = scaffold_total / N.
        #
        # Interpretation: "Among all compounds in our dataset, what fraction
        # contain this scaffold?" This becomes our null hypothesis - the rate
        # we'd expect if this taxon were a random sample of the dataset.
        #
        # "Enrichment" then means: "This taxon has MORE of this scaffold
        # than a random sample of the dataset would have."
        #
        # CAVEAT: Our dataset is NOT a random sample of all chemistry.
        # It's biased toward studied taxa (medicinal plants, crops, etc.).
        # So θ₀ = 10% might mean "10% of STUDIED compounds" not "10% of all
        # compounds in nature."
        #
        # WHY THIS IS OK: We're asking a RELATIVE question.
        # "Is Gentianaceae enriched in iridoids COMPARED TO our dataset?"
        # Not: "What's the true cosmic frequency of iridoids?"
        #
        # The weak prior (λ=1) further protects us: even if θ₀ is imperfect,
        # just 2-3 observations will dominate the posterior. The baseline
        # provides a rough reference point, not a strong constraint.
        #
        # OPTION 3: θ₀ = 0 (pure presence/absence)
        # ----------------------------------------
        # If θ₀ → 0, then P(θ > θ₀) → 1 for any scaffold present at all.
        # This loses discriminative power - everything is "enriched."
        #
        # CONCLUSION: Option 2 (observed frequency) is the best compromise.
        # It provides a meaningful baseline for "enrichment" while the weak
        # prior ensures data dominates quickly. The biases in θ₀ are
        # acceptable because we're asking relative, not absolute, questions.
        # ================================================================
        MIN_THETA_0 = cfg["stats"]["min_theta_0"]
        rank_data = rank_data.with_columns(
            [
                (pl.col("scaffold_total") / pl.col("N"))
                .clip(lower_bound=MIN_THETA_0)
                .alias("theta_0"),
            ],
        )

        # HIERARCHICAL PRIORS: Blend parent posterior with global baseline.
        #
        # The prior center is a weighted blend:
        #   prior_center = w * parent_mean + (1-w) * theta_0
        #
        # Where w = hierarchical_weight (0 = pure global, 1 = pure parent)
        #
        # This allows child's evidence to dominate while parent provides
        # subtle guidance. Default w=0.2 means prior is 80% global, 20% parent.
        if (
            cfg["priors"]["hierarchical_prior_flow"]
            and rank_idx > 0
            and rank in parent_child_maps
        ):
            parent_rank = valid_ranks[rank_idx - 1]
            parent_posteriors = posteriors_by_rank.get(parent_rank)

            if parent_posteriors is not None:
                child_to_parent = parent_child_maps[rank].rename(
                    {"child_taxon": "taxon_ancestor"},
                )
                rank_data_with_parent = rank_data.join(
                    child_to_parent,
                    on="taxon_ancestor",
                    how="left",
                ).join(
                    parent_posteriors.select(
                        [
                            "compound_ancestor",
                            "taxon_ancestor",
                            "posterior_mean",
                        ],
                    ).rename(
                        {
                            "taxon_ancestor": "parent_taxon",
                            "posterior_mean": "parent_mean",
                        },
                    ),
                    on=["compound_ancestor", "parent_taxon"],
                    how="left",
                )

                # ========================================================
                # PRIOR COMPUTATION
                # ========================================================
                #
                # The prior is our "belief before seeing data". It's a Beta
                # distribution centered on what we expect for this scaffold.
                #
                # THE PRIOR CENTER (what rate do we expect?):
                # -------------------------------------------
                # prior_center = w × parent_posterior + (1-w) × θ₀
                #
                # Where:
                # - θ₀ = scaffold's observed frequency in the dataset (imperfect
                #   baseline - see comments above for why "global frequency" is
                #   problematic with biased sampling)
                # - parent_posterior = parent taxon's estimated rate
                # - w = hierarchical_weight = 0.1 (10% parent, 90% θ₀)
                #
                # PRIOR STRENGTH (how confident in our prior?):
                # ---------------------------------------------
                # With λ=1, the prior is equivalent to having seen 1 pseudo-
                # observation at the prior_center rate. This is deliberately
                # weak so that even 2-3 real observations can override it.
                #
                # Example: θ₀ = 0.05 (5% of dataset), λ = 1
                # → α_prior = 0.05 × 1 = 0.05
                # → β_prior = 0.95 × 1 = 0.95
                # → Prior says "expect ~5%, but I'm not very sure"
                #
                # After seeing a=3 (3 compounds with scaffold), c_eff=7:
                # → α_post = 0.05 + 3 = 3.05
                # → β_post = 0.95 + 7 = 7.95
                # → Posterior ≈ 3.05/11 ≈ 28% (data dominates!)
                # ========================================================

                W = cfg["priors"]["hierarchical_weight"]
                LAMBDA_PRIOR = cfg["priors"]["prior_strength"]

                rank_data = (
                    rank_data_with_parent.with_columns(
                        [
                            # Blend parent posterior with global baseline for prior center
                            pl.when(pl.col("parent_mean").is_not_null())
                            .then(
                                W * pl.col("parent_mean") + (1 - W) * pl.col("theta_0"),
                            )
                            .otherwise(pl.col("theta_0"))
                            .alias("prior_center"),
                        ],
                    )
                    .with_columns(
                        [
                            (pl.col("prior_center") * LAMBDA_PRIOR).alias(
                                "alpha_prior",
                            ),
                            ((1 - pl.col("prior_center")) * LAMBDA_PRIOR).alias(
                                "beta_prior",
                            ),
                        ],
                    )
                    .drop(["parent_taxon", "parent_mean", "prior_center"])
                )
            else:
                # Coarsest rank or no parent: simple prior centered on θ₀
                LAMBDA_PRIOR = cfg["priors"]["prior_strength"]
                rank_data = rank_data.with_columns(
                    [
                        (pl.col("theta_0") * LAMBDA_PRIOR).alias("alpha_prior"),
                        ((1 - pl.col("theta_0")) * LAMBDA_PRIOR).alias("beta_prior"),
                    ],
                )
        else:
            # No hierarchical flow: simple prior centered on θ₀
            LAMBDA_PRIOR = cfg["priors"]["prior_strength"]
            rank_data = rank_data.with_columns(
                [
                    (pl.col("theta_0") * LAMBDA_PRIOR).alias("alpha_prior"),
                    ((1 - pl.col("theta_0")) * LAMBDA_PRIOR).alias("beta_prior"),
                ],
            )

        # Compute posterior
        rank_data = (
            rank_data.with_columns(
                [
                    (pl.col("alpha_prior") + pl.col("a")).alias("alpha_post"),
                    (pl.col("beta_prior") + pl.col("c_eff")).alias("beta_post"),
                ],
            )
            .with_columns(
                [
                    # Posterior mean: always well-defined
                    (
                        pl.col("alpha_post")
                        / (pl.col("alpha_post") + pl.col("beta_post"))
                    ).alias("posterior_mean"),
                    # Posterior mode: only defined when α > 1 AND β > 1
                    # Otherwise use the mean as a fallback
                    pl.when((pl.col("alpha_post") > 1) & (pl.col("beta_post") > 1))
                    .then(
                        (pl.col("alpha_post") - 1)
                        / (pl.col("alpha_post") + pl.col("beta_post") - 2),
                    )
                    .otherwise(
                        pl.col("alpha_post")
                        / (pl.col("alpha_post") + pl.col("beta_post")),
                    )
                    .clip(0, 1)
                    .alias("posterior_mode"),
                    # EFFECTIVE SAMPLE SIZE (ESS)
                    # ===========================
                    # ESS = α_post + β_post - (α_prior + β_prior)
                    # This measures how much the data contributed beyond the prior.
                    #
                    # Interpretation:
                    # - ESS = 0: No data, pure prior
                    # - ESS = 5: Equivalent to 5 observations
                    # - ESS = 20: Solid evidence
                    #
                    # PRINCIPLED CI FILTERING:
                    # Instead of arbitrary CI width thresholds, we can filter by ESS.
                    # ESS < 3 means fewer than 3 effective observations - unreliable.
                    (
                        pl.col("alpha_post")
                        + pl.col("beta_post")
                        - pl.col("alpha_prior")
                        - pl.col("beta_prior")
                    ).alias("effective_sample_size"),
                ],
            )
            .with_columns(
                [
                    # Observed rate (MLE): a_raw / taxon_total
                    # This is the actual observed proportion, before Bayesian shrinkage
                    (
                        pl.col("a_raw").cast(pl.Float32)
                        / (pl.col("taxon_total").cast(pl.Float32) + 1e-10)
                    ).alias("observed_rate"),
                    # log2 fold-change: posterior_mean vs baseline θ₀
                    (
                        (pl.col("posterior_mean") / (pl.col("theta_0") + 1e-10)).log(2)
                    ).alias("log2_enrichment"),
                ],
            )
        )

        # Posterior probability computation
        alpha_arr = rank_data.get_column("alpha_post").to_numpy()
        beta_arr = rank_data.get_column("beta_post").to_numpy()
        theta_arr = rank_data.get_column("theta_0").to_numpy()
        post_prob_arr = posterior_probability_above(alpha_arr, beta_arr, theta_arr)
        rank_data = rank_data.with_columns(
            [pl.Series("posterior_enrich_prob", post_prob_arr)],
        )

        posteriors_by_rank[rank] = rank_data
        logging.info(f"    {rank_data.height} associations")
        rank_results.append(rank_data)

    if not rank_results:
        return pl.DataFrame()

    result = pl.concat(rank_results, how="vertical")

    # ====================================================================
    # CREDIBLE INTERVAL AND ROPE COMPUTATION
    # ====================================================================

    logging.info("  Computing CI and ROPE decisions...")

    alpha_arr = result.get_column("alpha_post").to_numpy()
    beta_arr = result.get_column("beta_post").to_numpy()
    theta_arr = result.get_column("theta_0").to_numpy()
    CI_PROB = cfg["stats"]["ci_prob"]
    ROPE_HALF_WIDTH = cfg["stats"]["rope_half_width"]

    ci_lower_arr, ci_upper_arr = fold_change_credible_interval(
        alpha_arr,
        beta_arr,
        theta_arr,
        CI_PROB,
    )

    decisions_arr, p_above_arr, p_below_arr = rope_decision(
        alpha_arr,
        beta_arr,
        theta_arr,
        ROPE_HALF_WIDTH,
        CI_PROB,
    )

    # Add computed columns back to result
    result = result.with_columns(
        [
            pl.Series("ci_lower", ci_lower_arr),
            pl.Series("ci_upper", ci_upper_arr),
            pl.Series("rope_decision", decisions_arr),
            pl.Series("p_above_rope", p_above_arr),
            pl.Series("p_below_rope", p_below_arr),
        ],
    ).with_columns([(pl.col("ci_upper") - pl.col("ci_lower")).alias("ci_width")])

    # ====================================================================
    # STATISTICAL QUALITY FLAGS
    # ====================================================================
    # RELIABLE: ESS >= MIN_ESS (default 3)
    MIN_ESS = cfg["stats"]["min_ess"]
    result = result.with_columns(
        [
            (pl.col("effective_sample_size") >= MIN_ESS).alias("reliable"),
        ],
    )

    # Add taxon names
    result = result.join(
        taxon_name.rename({"taxon": "taxon_ancestor", "taxon_name": "taxon_name"}),
        on="taxon_ancestor",
        how="left",
    )

    # FINAL SORTING: Prioritize reliable results with high P(enrich)
    # Sort by: reliable (True first), then P(enrich) desc, then ESS desc, then IDs for determinism
    result = result.sort(
        [
            "reliable",
            "posterior_enrich_prob",
            "effective_sample_size",
            "compound_ancestor",
            "taxon_ancestor",
        ],
        descending=[True, True, True, False, False],
    )

    logging.info(f"Complete: {result.height} enrichments ({time.time() - start:.1f}s)")
    # Add human-readable rank labels to the result for all downstream displays.
    try:
        if "taxon_rank" in result.columns:
            rank_vals = result.get_column("taxon_rank").to_list()
            rank_labels = [get_rank_label(r) for r in rank_vals]
            result = result.with_columns(pl.Series("taxon_rank_label", rank_labels))
    except Exception:
        # Defensive: if anything goes wrong, don't break the analysis pipeline.
        pass

    return result


@app.function
def discover_top_taxa(
    markers: pl.DataFrame,
    rank: str,
    min_prob: float = MIN_PROB,
    min_ess: float = MIN_ESS,
    min_obs: float = MIN_OBS,
    min_log2fc: float = MIN_LOG2FC,
    top_n: int = TOP_N_TAXA,
) -> list[str]:
    """
    Select top N taxa with most distinctive chemistry at this rank.

    Returns list of taxon IDs (taxon_ancestor).
    """
    rank_markers = (
        markers.filter(pl.col("taxon_rank_label") == rank)
        .filter(pl.col("posterior_enrich_prob") >= min_prob)
        .filter(pl.col("effective_sample_size") >= min_ess)
        .filter(pl.col("log2_enrichment") >= min_log2fc)
        .filter(pl.col("a_raw") >= min_obs)
    )

    if rank_markers.is_empty():
        return []

    # Score taxa by chemical distinctiveness
    taxon_scores = (
        rank_markers.group_by("taxon_ancestor")
        .agg(
            [
                pl.col("compound_ancestor").n_unique().alias("n_scaffolds"),
                pl.col("posterior_enrich_prob").mean().alias("avg_prob"),
                pl.col("log2_enrichment").mean().alias("avg_fc"),
                pl.col("effective_sample_size").sum().alias("total_ess"),
            ]
        )
        .with_columns(
            [
                # Distinctiveness = diversity + strength + confidence
                (
                    pl.col("n_scaffolds").cast(pl.Float32) * 3.0
                    + pl.col("avg_fc") * 1.5
                    + pl.col("avg_prob") * 2.0
                    + (pl.col("total_ess") / 100.0).clip(0, 2.0)
                ).alias("distinctiveness")
            ]
        )
        .sort(["distinctiveness", "taxon_ancestor"], descending=[True, False])
        .head(top_n)
    )

    return taxon_scores.get_column("taxon_ancestor").to_list()


@app.function
def get_markers_for_top_taxa(
    markers: pl.DataFrame,
    rank: str,
    top_taxa: list[str],
    min_prob: float = MIN_PROB,
    min_ess: float = MIN_ESS,
    min_obs: float = MIN_OBS,
    min_log2fc: float = MIN_LOG2FC,
    top_n_per_taxon: int = TOP_N_MARKERS_PER_TAXON,
) -> pl.DataFrame:
    """
    Get top markers for each of the selected top taxa.
    """
    rank_markers = (
        markers.filter(pl.col("taxon_rank_label") == rank)
        .filter(pl.col("taxon_ancestor").is_in(top_taxa))
        .filter(pl.col("posterior_enrich_prob") >= min_prob)
        .filter(pl.col("effective_sample_size") >= min_ess)
        .filter(pl.col("log2_enrichment") >= min_log2fc)
        .filter(pl.col("a_raw") >= min_obs)
    )

    if rank_markers.is_empty():
        return pl.DataFrame()

    # For each taxon, get top N markers
    result = (
        rank_markers.with_columns(
            [
                # Marker quality score
                (
                    pl.col("posterior_enrich_prob") * 2.0
                    + pl.col("log2_enrichment") / 10.0
                    + (pl.col("effective_sample_size") / 100.0).clip(0, 1)
                ).alias("marker_quality")
            ]
        )
        .sort(["taxon_ancestor", "marker_quality"], descending=[False, True])
        .group_by("taxon_ancestor")
        .head(top_n_per_taxon)
    )

    return result


@app.function
def create_taxa_marker_heatmap(
    markers_df: pl.DataFrame,
    rank: str,
) -> alt.Chart:
    """
    Heatmap showing top 10 taxa (rows) × their markers (columns).
    """
    if markers_df.is_empty():
        return mo.md(f"*No {rank}-level data*")

    # Prepare data
    plot_data = markers_df.with_columns(
        [
            pl.col("compound_ancestor").alias("scaffold_short"),
        ]
    ).select(
        [
            "taxon_name",
            "scaffold_short",
            "log2_enrichment",
            "posterior_enrich_prob",
            "effective_sample_size",
            "a_raw",
        ]
    )

    # Get colormap
    try:
        import cmcrameri.cm as cmc

        cmap = cmc.batlow
        colors = [
            f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
            for c in [cmap(i / 10) for i in range(11)]
        ]
    except:
        colors = "viridis"

    chart = (
        alt.Chart(plot_data)
        .mark_rect(cornerRadius=2, stroke="white", strokeWidth=1)
        .encode(
            y=alt.Y(
                "taxon_name:N",
                title=None,
                sort=alt.EncodingSortField(
                    field="log2_enrichment", op="mean", order="descending"
                ),
                axis=alt.Axis(labelFontSize=11, labelFontWeight="bold"),
            ),
            x=alt.X(
                "scaffold_short:N",
                title=None,
                sort=alt.EncodingSortField(
                    field="log2_enrichment", op="max", order="descending"
                ),
                axis=alt.Axis(labelAngle=-45, labelFontSize=9),
            ),
            color=alt.Color(
                "log2_enrichment:Q",
                scale=alt.Scale(
                    range=colors if isinstance(colors, list) else "viridis",
                    scheme=colors if isinstance(colors, str) else "viridis",
                    domain=[0, 10],
                    clamp=True,
                ),
                legend=alt.Legend(title="log₂ FC", gradientLength=150),
            ),
            tooltip=[
                alt.Tooltip("taxon_name:N", title="Taxon"),
                alt.Tooltip("scaffold_short:N", title="Scaffold"),
                alt.Tooltip(
                    "posterior_enrich_prob:Q", format=".3f", title="P(enriched)"
                ),
                alt.Tooltip("log2_enrichment:Q", format=".2f", title="log₂ FC"),
                alt.Tooltip("effective_sample_size:Q", format=".1f", title="ESS"),
                alt.Tooltip("a_raw:Q", title="Observations"),
            ],
        )
        .properties(
            width=800,
            height=max(300, len(plot_data.get_column("taxon_name").unique()) * 40),
            title=alt.TitleParams(
                text=f"Top 10 at {rank} level and Their Chemical Markers",
                subtitle=f"Taxa selected by chemical distinctiveness (P≥{MIN_PROB}, ESS≥{MIN_ESS}, FC≥{MIN_LOG2FC})",
                fontSize=14,
                anchor="start",
            ),
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )

    return chart


@app.function
def create_taxa_summary_table(
    markers_df: pl.DataFrame,
    rank: str,
) -> pl.DataFrame:
    """
    Summary table: one row per taxon showing its marker statistics.
    """
    if markers_df.is_empty():
        return pl.DataFrame()

    return (
        markers_df.group_by("taxon_ancestor")
        .agg(
            [
                pl.col("taxon_name").first(),
                pl.col("compound_ancestor").n_unique().alias("n_markers"),
                pl.col("posterior_enrich_prob").mean().alias("avg_prob"),
                pl.col("log2_enrichment").mean().alias("avg_fc"),
                pl.col("log2_enrichment").max().alias("max_fc"),
                pl.col("effective_sample_size").sum().alias("total_ess"),
                pl.col("a_raw").sum().alias("total_obs"),
            ]
        )
        .sort("avg_prob", descending=True)
        .select(
            [
                pl.col("taxon_name").alias("Taxon"),
                pl.col("n_markers").alias("# Markers"),
                pl.col("avg_prob").round(3).alias("Avg P(enrich)"),
                pl.col("avg_fc").round(2).alias("Avg log₂FC"),
                pl.col("max_fc").round(2).alias("Max log₂FC"),
                pl.col("total_ess").round(1).alias("Total ESS"),
                pl.col("total_obs").alias("Total Obs"),
            ]
        )
    )


@app.function
def create_markers_detail_table(
    markers_df: pl.DataFrame,
    rank: str,
) -> pl.DataFrame:
    """
    Detailed table: one row per marker showing which taxa it's in.
    """
    if markers_df.is_empty():
        return pl.DataFrame()

    return (
        markers_df.sort(
            ["taxon_name", "posterior_enrich_prob"], descending=[False, True]
        )
        .select(
            [
                pl.col("taxon_name").alias("Taxon"),
                pl.col("compound_ancestor").alias("Scaffold"),
                pl.col("posterior_enrich_prob").round(3).alias("P(enrich)"),
                pl.col("log2_enrichment").round(2).alias("log₂FC"),
                pl.col("effective_sample_size").round(1).alias("ESS"),
                pl.col("a_raw").alias("Obs"),
            ]
        )
        .with_columns(
            [
                pl.col("Scaffold")
                .map_elements(lambda s: mo.image(svg_from_smiles(s)))
                .alias("Structure")
            ]
        )
    )


@app.cell
def header():
    mo.md("""
    # Bayesian Chemotaxonomic Scaffold Discovery

    Discover chemical scaffolds enriched in taxonomic groups using Bayesian inference.
    """)
    return


@app.cell
def apply_config():
    effective_config = DEFAULT_CONFIG
    return (effective_config,)


@app.cell
def load_data(effective_config):
    """Load compound, scaffold, and taxonomy data from Wikidata via SPARQL."""

    with mo.status.spinner("Fetching data from Wikidata..."):
        logging.info("=" * 55)
        logging.info("Loading data from Wikidata SPARQL endpoint")
        logging.info("=" * 55)

        # Fetch compound SMILES from Wikidata (compound ID -> SMILES mapping)
        compound_smiles = parse_sparql_response(
            execute_with_retry(
                query="""
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                                    SELECT DISTINCT ?compound ?compound_smiles WHERE {
                ?compound wdt:P233 ?compound_smiles .
                }
                """,
                endpoint=effective_config["qlever_endpoint"],
            )
        )
        logging.info(f"✓ Compound SMILES: {compound_smiles.height:,} compounds")

        # Load canonical SMILES from local file - REQUIRED for MORTAR row mapping
        # The MORTAR fragment files map by row order to this file
        compound_can_smiles = pl.read_csv(
            effective_config["data_paths"]["path_can_smi"],
            has_header=False,
            new_columns=["compound_smiles"],
        )
        logging.info(
            f"✓ Canonical SMILES (local): {compound_can_smiles.height:,} compounds",
        )

        # Try to load local fragment files if they exist
        items_path = select_first_existing(
            effective_config["data_paths"],
            ["path_items_cdk", "path_items_ert", "path_items_sru"],
        )
        compound_mapping = (
            load_compound_fragment_mapping(items_path) if items_path else None
        )

        frag_tables = []
        for attr in ["path_frags_cdk", "path_frags_ert", "path_frags_sru"]:
            p = Path(effective_config["data_paths"][attr])
            frag_tables.append(
                load_fragments(
                    p, effective_config["filtering"]["min_frequency_scaffold"]
                ),
            )

        scaffold_fragments = (
            pl.concat(frag_tables).unique(subset=["SMILES"])
            if frag_tables
            else pl.DataFrame({"SMILES": [], "MoleculeFrequency": []})
        )
        logging.info(
            f"✓ Scaffold fragments: {scaffold_fragments.height:,} (≥{effective_config["filtering"]["min_frequency_scaffold"]} occurrences)",
        )

        scaffolds_base = build_compound_scaffold_table(
            compound_can_smiles,
            compound_mapping,
            scaffold_fragments,
        )
        logging.info(
            f"✓ Base scaffolds: {scaffolds_base.height:,} compound-scaffold pairs",
        )

        if effective_config["filtering"]["include_compounds_as_scaffolds"]:
            compound_self = compound_can_smiles.select(
                [
                    pl.col("compound_smiles"),
                    pl.col("compound_smiles").alias("scaffold"),
                ],
            )
            scaffolds_base = pl.concat([scaffolds_base, compound_self]).unique()
            logging.info(
                f"✓ Including compounds as scaffolds: +{compound_self.height:,} whole molecules",
            )

        compound_scaffold = compound_smiles.join(
            scaffolds_base,
            on="compound_smiles",
            how="inner",
        )
        logging.info(f"✓ Total compound-scaffold pairs: {compound_scaffold.height:,}")

        # Fetch taxon data from Wikidata
        taxon_name = parse_sparql_response(
            execute_with_retry(
                query="""
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                SELECT DISTINCT ?taxon ?taxon_name WHERE {
                  ?taxon wdt:P225 ?taxon_name .
                }
                """,
                endpoint=effective_config["qlever_endpoint"],
            )
        )
        logging.info(f"✓ Taxon names: {taxon_name.height:,}")

        taxon_parent = parse_sparql_response(
            execute_with_retry(
                query="""
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                SELECT DISTINCT ?taxon ?taxon_parent WHERE {
                  ?taxon wdt:P171 ?taxon_parent .
                }
                """,
                endpoint=effective_config["qlever_endpoint"],
            )
        )
        logging.info(f"✓ Taxon hierarchy: {taxon_parent.height:,} parent relationships")

        taxon_rank = parse_sparql_response(
            execute_with_retry(
                query="""
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                SELECT DISTINCT ?taxon ?taxon_rank WHERE {
                  ?taxon wdt:P105 ?taxon_rank .
                }
                """,
                endpoint=effective_config["qlever_endpoint"],
            )
        )
        logging.info(f"✓ Taxon ranks: {taxon_rank.height:,}")

        compound_taxon = parse_sparql_response(
            execute_with_retry(
                query="""
                PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                SELECT DISTINCT ?compound ?taxon WHERE {
                  ?compound wdt:P703 ?taxon .
                }
                """,
                endpoint=effective_config["qlever_endpoint"],
            )
        )
        logging.info(f"✓ Compound-taxon annotations: {compound_taxon.height:,}")
        logging.info("=" * 55)
    return (
        compound_scaffold,
        compound_taxon,
        taxon_name,
        taxon_parent,
        taxon_rank,
    )


@app.cell
def build_lineages(
    compound_scaffold,
    compound_taxon,
    taxon_parent,
    taxon_rank,
):
    with mo.status.spinner("Building taxonomic lineages..."):
        compound_lineage = build_compound_lineage(compound_scaffold)
        taxon_lineage = build_taxon_lineage(
            taxon_parent,
            taxon_rank,
            compound_taxon.get_column("taxon"),
        )
    return compound_lineage, taxon_lineage


@app.cell
def lineage_diagnostics(
    compound_lineage,
    compound_taxon,
    taxon_lineage,
    taxon_rank,
):
    with mo.status.spinner("Computing lineage diagnostics..."):
        ld_null_rank_count = taxon_lineage.filter(pl.col("taxon_rank").is_null()).height
        ld_total_lineage_rows = taxon_lineage.height
        ld_null_rank_pct = (
            100 * ld_null_rank_count / ld_total_lineage_rows
            if ld_total_lineage_rows > 0
            else 0
        )

        ld_available_ranks = taxon_rank.get_column("taxon_rank").unique().to_list()
        ld_available_ranks_sorted = sorted(
            ld_available_ranks,
            key=lambda r: get_rank_order(r),
        )

        ld_base_evidence = (
            compound_taxon.join(compound_lineage, on="compound", how="inner")
            .select(["compound", "taxon", "compound_ancestor"])
            .unique()
        )

        ld_rank_compound_counts = []
        for ld_rank in ld_available_ranks_sorted:
            ld_rank_map = (
                taxon_lineage.filter(pl.col("taxon_rank") == ld_rank)
                .select(["taxon", "taxon_ancestor"])
                .unique()
            )
            ld_rank_evidence = (
                ld_base_evidence.join(ld_rank_map, on="taxon", how="inner")
                .select(["compound", "taxon_ancestor"])
                .unique()
            )
            ld_n_compounds = (
                ld_rank_evidence.get_column("compound").n_unique()
                if not ld_rank_evidence.is_empty()
                else 0
            )
            ld_n_taxa = (
                ld_rank_evidence.get_column("taxon_ancestor").n_unique()
                if not ld_rank_evidence.is_empty()
                else 0
            )
            ld_rank_compound_counts.append(
                {
                    "rank": ld_rank,
                    "rank_label": get_rank_label(ld_rank),
                    "n_mappings": ld_rank_map.height,
                    "n_compounds_mapped": ld_n_compounds,
                    "n_taxa_at_rank": ld_n_taxa,
                },
            )

        ld_diag_df = pl.DataFrame(ld_rank_compound_counts).filter(
            pl.col(name="n_compounds_mapped") > 0,
        )

    _out = mo.vstack(
        [
            mo.md("### Taxonomic Lineage Coverage"),
            mo.md(
                f"**Coverage:** {ld_total_lineage_rows - ld_null_rank_count:,} / {ld_total_lineage_rows:,} entries have assigned ranks ({100 - ld_null_rank_pct:.1f}%)",
            ),
            mo.md("**Compounds mapped per rank:**"),
            mo.ui.table(
                ld_diag_df.select(
                    [
                        pl.col("rank_label").alias("Rank"),
                        pl.col("n_taxa_at_rank").alias("Taxa"),
                        pl.col("n_compounds_mapped").alias("Compounds"),
                    ],
                ),
                selection=None,
            ),
        ],
    )
    _out
    return


@app.cell
def scaffold_compound_stats(compound_scaffold):
    with mo.status.spinner("Analyzing fragment coverage..."):
        # Filter to fragments only (scaffold != compound)
        scs_fragments = compound_scaffold.filter(
            pl.col("scaffold") != pl.col("compound_smiles"),
        )

        # Count compounds per fragment
        scs_counts = (
            scs_fragments.group_by("scaffold")
            .agg(pl.col("compound_smiles").n_unique().alias("n_compounds"))
            .sort("n_compounds", descending=True)
        )

        scs_total = scs_counts.height
        scs_median = scs_counts.get_column("n_compounds").median()
        scs_max = scs_counts.get_column("n_compounds").max()

        # Build table with thumbnails in one pipeline
        scs_df = (
            scs_counts.with_columns(
                [
                    pl.col("scaffold")
                    .map_elements(lambda s: mo.image(svg_from_smiles(s)))
                    .alias("Structure")
                ]
            )
            .select(["Structure", "n_compounds"])
            .rename({"n_compounds": "Compounds"})
        )

    _out = mo.vstack(
        [
            mo.md("### Chemical Fragment Coverage"),
            mo.md(
                """
            For now, scaffolds and fragments come from [MORTAR](https://github.com/FelixBaensch/MORTAR) (which requires a local installation and local files), but the long-term goal is to pull everything directly from Wikidata.
            """
            ).callout(
                kind="info",
            ),
            mo.md(
                f"**{scs_total:,}** unique fragments | Median: **{scs_median:.0f}** compounds/fragment | Max: **{scs_max}**",
            ),
            mo.ui.table(scs_df, selection=None),
        ],
    )
    _out
    return


@app.cell
def scaffold_trace_diagnostic(markers):
    st_available_ranks = sorted(
        markers.get_column("taxon_rank").unique().to_list(),
        key=lambda r: get_rank_order(r),
    )

    # Sort by max posterior, then scaffold size for reproducibility on ties
    st_top_scaffolds = (
        markers.group_by("compound_ancestor")
        .agg([pl.col("posterior_enrich_prob").max().alias("max_p")])
        .with_columns(
            pl.col("compound_ancestor").str.len_chars().alias("scaffold_size"),
        )
        .sort(["max_p", "scaffold_size"], descending=[True, True])
        .head(5)
    )

    trace_results = []
    for st_row in st_top_scaffolds.iter_rows(named=True):
        st_scaffold = st_row["compound_ancestor"]
        st_scaffold_markers = markers.filter(pl.col("compound_ancestor") == st_scaffold)
        for st_rank in st_available_ranks:
            st_rank_data = st_scaffold_markers.filter(pl.col("taxon_rank") == st_rank)
            if not st_rank_data.is_empty():
                st_best = st_rank_data.sort(
                    ["posterior_enrich_prob", "taxon_ancestor"],
                    descending=[True, False],
                ).row(0, named=True)
                st_decision = st_best.get("rope_decision", "?")
                st_badge = (
                    "🟢"
                    if st_decision == "enriched"
                    else ("🔴" if st_decision == "depleted" else "🟡")
                )
                trace_results.append(
                    {
                        "Scaffold": st_scaffold,
                        "Rank": get_rank_label(st_rank),
                        "Taxon": st_best.get("taxon_name", "?"),
                        "n": st_best.get("a_raw", 0),
                        "FC": round(st_best.get("log2_enrichment", 0), 1),
                        "Status": st_badge,
                    },
                )

    # Filter out very weak evidence rows (n <= 1) which are not informative
    trace_df = (
        pl.DataFrame(trace_results).filter(pl.col("n") > 1)
        if trace_results
        else pl.DataFrame()
    )

    # Add a small legend explaining status badges used in the table
    legend_md = mo.md(
        "**Legend:** 🟢 enriched | 🔴 depleted | 🟡 undecided | (rows with n ≤ 1 removed)"
    )

    _out = mo.vstack(
        [
            mo.md("### Top Scaffolds by Rank"),
            mo.ui.table(trace_df, selection=None),
            legend_md if not trace_df.is_empty() else mo.md("*No data*"),
        ],
    )
    _out
    return


@app.cell
def run_analysis(
    compound_lineage,
    compound_taxon,
    effective_config,
    taxon_lineage,
    taxon_name,
    taxon_rank,
):
    with mo.status.spinner("Computing Bayesian enrichment analysis..."):
        markers = run_hierarchical_analysis(
            compound_taxon,
            compound_lineage,
            taxon_lineage,
            taxon_name,
            taxon_rank,
            effective_config,
        )
    return (markers,)


@app.cell
def discovery_mode(effective_config, markers):
    # Sort by: ROPE decision (enriched first), then ESS desc
    # Custom sort: enriched > undecided > equivalent > depleted
    CI_PROB = effective_config["stats"]["ci_prob"]
    MIN_ESS = effective_config["stats"]["min_ess"]
    dm_top = (
        markers.with_columns(
            [
                pl.when(pl.col("rope_decision") == "enriched")
                .then(pl.lit(0))
                .when(pl.col("rope_decision") == "undecided")
                .then(pl.lit(1))
                .when(pl.col("rope_decision") == "equivalent")
                .then(pl.lit(2))
                .otherwise(pl.lit(3))
                .alias("_sort_decision"),
            ],
        )
        .sort(
            [
                "_sort_decision",
                "reliable",
                "effective_sample_size",
                "compound_ancestor",
                "taxon_ancestor",
            ],
            descending=[False, True, True, False, False],
        )
        .drop("_sort_decision")
    )

    dm_display = dm_top.select(
        [
            pl.col("compound_ancestor").alias("Scaffold"),
            pl.col("taxon_name").fill_null(pl.col("taxon_ancestor")).alias("Taxon"),
            # Use taxon_rank_label if present, otherwise fall back to raw taxon_rank
            pl.coalesce([pl.col("taxon_rank_label"), pl.col("taxon_rank")]).alias(
                "Rank"
            ),
            pl.col("a_raw").alias("n"),
            pl.col("n_source_taxa").alias("Taxa"),
            pl.col("effective_sample_size").round(1).alias("ESS"),
            pl.col("log2_enrichment").round(2).alias("log₂FC"),
            (
                pl.lit("[")
                + pl.col("ci_lower").round(2).cast(pl.Utf8)
                + pl.lit(", ")
                + pl.col("ci_upper").round(2).cast(pl.Utf8)
                + pl.lit("]")
            ).alias(f"{int(CI_PROB * 100)}% CI"),
            pl.col("rope_decision").alias("Decision"),
            pl.when(pl.col("reliable"))
            .then(pl.lit("✓"))
            .otherwise(pl.lit("○"))
            .alias("Rel"),
        ],
    )

    # Summary stats
    dm_n_enriched = markers.filter(pl.col("rope_decision") == "enriched").height
    dm_n_reliable = markers.filter(
        pl.col("reliable") & (pl.col("rope_decision") == "enriched"),
    ).height

    _out = mo.vstack(
        [
            mo.md(f"""
    ## All Enrichment Results

    🟢 **{dm_n_enriched:,}** enriched | ✓ **{dm_n_reliable:,}** reliable (ESS ≥ {MIN_ESS}) | **{dm_top.height:,}** total
    """),
            mo.ui.table(dm_display, selection=None),
        ],
    )
    _out
    return (MIN_ESS,)


@app.cell
def summary_stats(compound_scaffold, markers):
    ss_n_total = markers.height
    ss_n_scaffolds = markers.get_column("compound_ancestor").n_unique()
    ss_ranks = markers.get_column("taxon_rank").unique().to_list()
    ss_rank_labels = [get_rank_label(r) for r in sorted(ss_ranks, key=get_rank_order)]

    # Calculate scaffold type breakdown
    ss_unique_scaffolds = markers.get_column("compound_ancestor").unique()
    ss_unique_compounds = compound_scaffold.get_column("compound_smiles").unique()

    ss_n_compound = len(
        set(ss_unique_scaffolds.to_list()) & set(ss_unique_compounds.to_list()),
    )
    ss_n_fragment = ss_n_scaffolds - ss_n_compound

    _out = mo.vstack(
        [
            mo.md("### Analysis Summary"),
            mo.hstack(
                [
                    mo.stat(
                        value=str(ss_n_total),
                        label="Enrichment Pairs",
                        caption="scaffold × taxon",
                    ),
                    mo.stat(
                        value=str(ss_n_scaffolds),
                        label="Unique Scaffolds",
                        caption=f"{ss_n_compound} compounds + {ss_n_fragment} fragments",
                    ),
                    mo.stat(
                        value=str(len(ss_ranks)),
                        label="Taxonomic Ranks",
                        caption=", ".join(ss_rank_labels),
                    ),
                ],
                justify="start",
                wrap=True,
                gap=2,
            ),
        ],
    )
    _out
    return


@app.cell
def lineage_profile_viz(markers):
    """Scaffold selector for lineage profile visualization."""

    mo.md("---")

    # Find "good examples" - scaffolds with consistent enrichment across ranks
    # Sort by average enrichment (best signal), then scaffold size for reproducibility
    good_examples_df = (
        markers.filter(pl.col("rope_decision") == "enriched")
        .group_by("compound_ancestor")
        .agg(
            [
                pl.col("taxon_rank").n_unique().alias("n_ranks"),
                pl.col("log2_enrichment").mean().alias("avg_fc"),
            ],
        )
        .filter(pl.col("n_ranks") >= 3)  # Enriched in at least 3 ranks
        .with_columns(
            pl.col("compound_ancestor").str.len_chars().alias("scaffold_size"),
        )
        .sort(["avg_fc", "scaffold_size"], descending=[True, True])
    )
    good_examples = good_examples_df.get_column("compound_ancestor").to_list()

    # Fall back to all scaffolds if no good examples
    # Sort by max posterior, then scaffold size for reproducibility
    if not good_examples:
        good_examples = (
            markers.group_by("compound_ancestor")
            .agg(pl.col("posterior_enrich_prob").max().alias("max_p"))
            .with_columns(
                pl.col("compound_ancestor").str.len_chars().alias("scaffold_size"),
            )
            .sort(["max_p", "scaffold_size"], descending=[True, True])
            .get_column("compound_ancestor")
            .to_list()
        )

    scaffold_select = mo.ui.dropdown(
        options=good_examples if good_examples else ["No scaffolds"],
        value=good_examples[0] if good_examples else None,
        label="Select scaffold (ranked by size, then enrichment)",
    )
    scaffold_select
    return (scaffold_select,)


@app.cell
def show_lineage_profile(markers, scaffold_select, taxon_lineage):
    selected_scaffold = scaffold_select.value

    if not selected_scaffold or markers.is_empty():
        _out = mo.md("*Select a scaffold to view its lineage profile*")
    else:
        lp_profile = markers.filter(pl.col("compound_ancestor") == selected_scaffold)

        if lp_profile.is_empty():
            _out = mo.md(f"*No data found for scaffold: {selected_scaffold}*")
        else:
            LP_RANKS = [
                "Kingdom",
                "Phylum",
                "Class",
                "Order",
                "Family",
                "Genus",
                "Species",
            ]

            # Create lookup table for rank labels
            unique_ranks = lp_profile.get_column("taxon_rank").unique().to_list()
            rank_lookup = pl.DataFrame(
                {
                    "taxon_rank": unique_ranks,
                    "lp_rank_label": [get_rank_label(r) for r in unique_ranks],
                    "lp_rank_order": [get_rank_order(r) for r in unique_ranks],
                },
            )

            # Add rank labels via join
            lp_profile_with_labels = lp_profile.join(
                rank_lookup,
                on="taxon_rank",
                how="left",
            )

            # Filter to standard ranks only
            lp_profile_named = lp_profile_with_labels.filter(
                pl.col("lp_rank_label").is_in(LP_RANKS),
            )

            if lp_profile_named.is_empty():
                # Fallback: show top enrichments without rank filtering
                lp_top = lp_profile.sort(
                    ["posterior_enrich_prob", "taxon_ancestor"],
                    descending=[True, False],
                ).head(5)
                lp_summary_rows = []
                for lp_row in lp_top.iter_rows(named=True):
                    lp_summary_rows.append(
                        f"- {lp_row.get('taxon_name', '?')}: {lp_row.get('posterior_enrich_prob', 0):.3f}",
                    )
                _out = mo.md(f"""
                ### Lineage Profile: `{selected_scaffold}`

                *No standard rank data found. Top enrichments:*

                {chr(10).join(lp_summary_rows)}
                """)
            else:
                # FIND MAIN LINEAGE - best average posterior across ranks
                lp_species = lp_profile_named.filter(
                    pl.col("lp_rank_label") == "Species",
                )

                lp_best_lineage_score = 0.0
                lp_best_lineage_taxa = set()

                if not lp_species.is_empty():
                    for sp_row in lp_species.iter_rows(named=True):
                        sp_taxon = sp_row["taxon_ancestor"]
                        sp_ancestors = set(
                            taxon_lineage.filter(pl.col("taxon") == sp_taxon)
                            .get_column("taxon_ancestor")
                            .to_list(),
                        ) | {sp_taxon}

                        lineage_data = lp_profile_named.filter(
                            pl.col("taxon_ancestor").is_in(list(sp_ancestors)),
                        )

                        if len(lineage_data) > 0:
                            # Sum posterior across lineage (higher = better)
                            total_posterior = (
                                lineage_data.get_column("posterior_enrich_prob").sum()
                                or 0
                            )
                            if total_posterior > lp_best_lineage_score:
                                lp_best_lineage_score = total_posterior
                                lp_best_lineage_taxa = sp_ancestors

                # Fallback: if no species, use best overall taxon
                if not lp_best_lineage_taxa:
                    lp_best_overall = lp_profile_named.sort(
                        ["posterior_enrich_prob", "taxon_ancestor"],
                        descending=[True, False],
                    ).head(1)
                    if not lp_best_overall.is_empty():
                        lp_best_taxon = lp_best_overall.get_column("taxon_ancestor")[0]
                        lp_best_lineage_taxa = set(
                            taxon_lineage.filter(pl.col("taxon") == lp_best_taxon)
                            .get_column("taxon_ancestor")
                            .to_list(),
                        ) | {lp_best_taxon}

                # For each rank, get the main lineage taxon
                lp_main_by_rank = {}
                for lp_rank in LP_RANKS:
                    lp_rank_main = lp_profile_named.filter(
                        (pl.col("lp_rank_label") == lp_rank)
                        & (pl.col("taxon_ancestor").is_in(list(lp_best_lineage_taxa))),
                    ).sort(
                        ["posterior_enrich_prob", "taxon_ancestor"],
                        descending=[True, False],
                    )

                    if not lp_rank_main.is_empty():
                        lp_main_by_rank[lp_rank] = lp_rank_main.row(0, named=True)

                # Build table data for clean display
                lp_table_data = []
                lp_divergent_count = 0

                for lp_rank in LP_RANKS:
                    lp_rank_data = lp_profile_named.filter(
                        pl.col("lp_rank_label") == lp_rank,
                    )

                    if lp_rank_data.is_empty():
                        continue  # Skip empty ranks

                    lp_m = lp_main_by_rank.get(lp_rank)

                    if lp_m:
                        lp_decision = lp_m.get("rope_decision", "?")

                        # Only show enriched or depleted, skip undecided/equivalent
                        if lp_decision not in ("enriched", "depleted"):
                            continue

                        lp_taxon = str(lp_m.get("taxon_name", "?"))
                        lp_fc = lp_m.get("log2_enrichment", 0)
                        lp_n = lp_m.get("a_raw", 0)

                        status = "🟢" if lp_decision == "enriched" else "🔴"

                        lp_table_data.append(
                            {
                                "Rank": lp_rank,
                                "Taxon": lp_taxon,
                                "FC": round(lp_fc, 1),
                                "n": lp_n,
                                "Status": status,
                            },
                        )

                    # Check for divergent taxa (only enriched)
                    lp_div_rows = lp_rank_data.filter(
                        ~pl.col("taxon_ancestor").is_in(list(lp_best_lineage_taxa))
                        & (pl.col("rope_decision") == "enriched"),
                    ).sort(
                        ["effective_sample_size", "taxon_ancestor"],
                        descending=[True, False],
                    )

                    for lp_div in lp_div_rows.head(1).iter_rows(named=True):
                        lp_div_taxon = str(lp_div.get("taxon_name", "?"))
                        lp_div_fc = lp_div.get("log2_enrichment", 0)
                        lp_table_data.append(
                            {
                                "Rank": "  ↳ other",
                                "Taxon": lp_div_taxon,
                                "FC": round(lp_div_fc, 1),
                                "n": lp_div.get("a_raw", 0),
                                "Status": "🔶",
                            },
                        )
                        lp_divergent_count += 1

                lp_df = pl.DataFrame(lp_table_data)

                # Pattern summary
                if lp_df.is_empty():
                    lp_pattern = mo.callout(
                        "No significant enrichment found",
                        kind="info",
                    )
                elif lp_divergent_count >= 3:
                    lp_pattern = mo.callout(
                        "Found in multiple unrelated lineages — possible convergent evolution",
                        kind="warn",
                    )
                elif lp_divergent_count >= 1:
                    lp_pattern = mo.callout(
                        "Some enrichment outside main lineage",
                        kind="info",
                    )
                else:
                    lp_pattern = mo.callout(
                        "Enrichment follows single lineage — strong phylogenetic signal",
                        kind="success",
                    )

                # Structure image
                lp_struct_img = mo.image(svg_from_smiles([selected_scaffold][0]))
                lp_short_smi = selected_scaffold

                _out = mo.vstack(
                    [
                        mo.md("### Taxonomic Lineage Profile"),
                        mo.vstack(
                            [lp_struct_img, mo.md(f"`{lp_short_smi}`")],
                            justify="start",
                            gap=1,
                        ),
                        mo.ui.table(lp_df, selection=None)
                        if not lp_df.is_empty()
                        else mo.md("*No enriched ranks*"),
                        lp_pattern,
                    ],
                )
    _out
    return


@app.cell
def top10_md():
    _out = mo.vstack(
        [
            mo.md("# Top 10 Taxa by Chemical Distinctiveness"),
            mo.md("""
        For each rank, we find the **10 taxa with most distinctive chemistry**, 
        then show all their enriched scaffolds.

        **Selection criteria:**
        - P(enriched) ≥ 0.89 (high confidence)
        - ESS ≥ 3 (reliable data)
        - OBS ≥ 3 (reliable data)
        - log₂FC ≥ 1.5 (strong enrichment, ≥3× baseline)
        """),
        ]
    )
    _out
    return


@app.cell
def markers_kin(
    markers,
):
    kingdom_top_taxa = discover_top_taxa(markers, "Kingdom")
    kingdom_markers = get_markers_for_top_taxa(markers, "Kingdom", kingdom_top_taxa)
    kingdom_summary = create_taxa_summary_table(kingdom_markers, "Kingdom")
    kingdom_detail = create_markers_detail_table(kingdom_markers, "Kingdom")

    _out = mo.vstack(
        [
            mo.md("## Kingdom Level"),
            create_taxa_marker_heatmap(kingdom_markers, "Kingdom"),
            mo.md("### Summary by Kingdom"),
            mo.ui.table(kingdom_summary, selection=None)
            if not kingdom_summary.is_empty()
            else mo.md("*No data*"),
            mo.md("### Detailed Markers"),
            mo.ui.table(kingdom_detail, selection=None)
            if not kingdom_detail.is_empty()
            else mo.md("*No data*"),
        ]
    )

    _out
    return


@app.cell
def markers_fam(
    markers,
):
    family_top_taxa = discover_top_taxa(markers, "Family")
    family_markers = get_markers_for_top_taxa(markers, "Family", family_top_taxa)
    family_summary = create_taxa_summary_table(family_markers, "Family")
    family_detail = create_markers_detail_table(family_markers, "Family")

    _out = mo.vstack(
        [
            mo.md("## Family Level"),
            create_taxa_marker_heatmap(family_markers, "Family"),
            mo.md("### Summary by Family"),
            mo.ui.table(family_summary, selection=None)
            if not family_summary.is_empty()
            else mo.md("*No data*"),
            mo.md("### Detailed Markers"),
            mo.ui.table(family_detail, selection=None)
            if not family_detail.is_empty()
            else mo.md("*No data*"),
        ]
    )

    _out
    return


@app.cell
def markers_gen(
    markers,
):
    genus_top_taxa = discover_top_taxa(markers, "Genus")
    genus_markers = get_markers_for_top_taxa(markers, "Genus", genus_top_taxa)
    genus_summary = create_taxa_summary_table(genus_markers, "Genus")
    genus_detail = create_markers_detail_table(genus_markers, "Genus")

    _out = mo.vstack(
        [
            mo.md("## Genus Level"),
            create_taxa_marker_heatmap(genus_markers, "Genus"),
            mo.md("### Summary by Genus"),
            mo.ui.table(genus_summary, selection=None)
            if not genus_summary.is_empty()
            else mo.md("*No data*"),
            mo.md("### Detailed Markers"),
            mo.ui.table(genus_detail, selection=None)
            if not genus_detail.is_empty()
            else mo.md("*No data*"),
        ]
    )

    _out
    return


@app.cell
def methods_summary(
    compound_scaffold,
    compound_taxon,
    effective_config,
    markers,
):
    """Statistical methods summary for reproducibility and publication."""

    # Count key statistics
    n_compounds = compound_scaffold.get_column("compound_smiles").n_unique()
    n_taxa_s = compound_taxon.get_column("taxon").n_unique()
    n_pairs = markers.height
    n_enriched = markers.filter(pl.col("rope_decision") == "enriched").height
    n_reliable = markers.filter(
        pl.col("reliable") & (pl.col("rope_decision") == "enriched"),
    ).height

    _out = mo.vstack(
        [
            mo.md("---"),
            mo.md("### Methods Summary"),
            mo.md(f"""
    **Data:** {n_compounds:,} compounds across {n_taxa_s:,} taxa

    **Statistical Model:**
    - Prior: Beta(θ₀λ, (1-θ₀)λ) with λ = {effective_config["priors"]["prior_strength"]}
    - Baseline θ₀: Scaffold's observed frequency in dataset
    - Diversity weighting: a_eff = √(compounds × source taxa)
    - Hierarchical prior flow: {effective_config["priors"]["hierarchical_prior_flow"]} (weight = {effective_config["priors"]["hierarchical_weight"]})

    **Decision Framework:**
    - Credible interval: {int(effective_config["stats"]["ci_prob"] * 100)}%
    - ROPE half-width: ±{effective_config["stats"]["rope_half_width"]} log₂ FC
    - Reliable if ESS ≥ {effective_config["stats"]["min_ess"]}

    **Results:** {n_pairs:,} scaffold-taxon pairs tested  → {n_enriched:,} enriched ({n_reliable:,} reliable)
    """),
        ],
    )
    _out
    return


@app.cell
def footer():
    mo.md("""
    ---
    **Data:**
    <a href="https://www.wikidata.org/wiki/Q104225190" style="color:#990000;">LOTUS Initiative</a> &
    <a href="https://www.wikidata.org/" style="color:#990000;">Wikidata</a> |
    **Code:**
    <a href="https://github.com/Adafede/marimo/blob/main/apps/bayesian_chemotaxonomic_profiler.py" style="color:#339966;">bayesian_chemotaxonomic_profiler.py</a> |
    **External tools:**
    <a href="https://github.com/FelixBaensch/MORTAR" style="color:#006699;">MORTAR</a> &
    <a href="https://github.com/cdk/depict" style="color:#006699;">CDK Depict</a> &
    <a href="https://qlever.dev/wikidata" style="color:#006699;">QLever</a> |
    **License:**
    <a href="https://creativecommons.org/publicdomain/zero/1.0/" style="color:#484848;">CC0 1.0</a> for data &
    <a href="https://www.gnu.org/licenses/agpl-3.0.html" style="color:#484848;">AGPL-3.0</a> for code
    """)
    return


if __name__ == "__main__":
    app.run()
