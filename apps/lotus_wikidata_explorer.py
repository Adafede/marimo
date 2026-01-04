# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "httpx==0.28.1",
#     "marimo",
#     "polars==1.35.2",
#     "pyarrow==22.0.0",
#     "rdflib==7.4.0",
#     "sparqlx==0.3.0",
# ]
# [tool.marimo.display]
# theme = "system"
# ///

"""
LOTUS Wikidata Explorer

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

__generated_with = "0.18.1"
app = marimo.App(width="full", app_title="LOTUS Wikidata Explorer")

with app.setup:
    import marimo as mo
    import time
    from datetime import datetime

    # Import modules
    from lotus_wikidata_explorer_modules.core import CONFIG
    from lotus_wikidata_explorer_modules.core.utils import extract_qid
    from lotus_wikidata_explorer_modules.ui import (
        parse_url_state,
        should_auto_run,
        format_url_params_display,
        resolve_taxon_to_qid,
        create_link,
        create_wikidata_link,
        create_structure_image_url,
        pluralize,
        build_api_url,
        build_filters_dict,
        get_filter_values,
    )
    from lotus_wikidata_explorer_modules.data import (
        query_wikidata,
        prepare_export_dataframe,
        create_formula_filters,
    )
    from lotus_wikidata_explorer_modules.export import (
        create_export_metadata,
        create_dataset_hashes,
        create_citation_text,
        export_to_rdf_turtle,
        generate_filename,
        compress_if_large,
    )

    # Set output max bytes safely
    try:
        mo._runtime.context.get_context().marimo_config["runtime"][
            "output_max_bytes"
        ] = 1_000_000_000  # 1GB for large datasets
    except Exception:
        pass


@app.cell
def parse_state(mo, parse_url_state, should_auto_run):
    """Parse URL parameters into application state."""
    url_params = mo.query_params()
    state = parse_url_state(url_params)
    state_auto_run = should_auto_run(url_params)

    # Extract individual state values for UI binding
    state_taxon = state.get("taxon", "")
    state_smiles = state.get("smiles", "")
    state_smiles_search_type = state.get("smiles_search_type", "substructure")
    state_smiles_threshold = state.get("smiles_threshold", 0.8)

    state_mass_filter = state.get("mass_filter", False)
    state_mass_min = state.get("mass_min", 0)
    state_mass_max = state.get("mass_max", 2000)

    state_year_filter = state.get("year_filter", False)
    state_year_start = state.get("year_start", 1900)
    state_year_end = state.get("year_end", None)

    state_formula_filter = state.get("formula_filter", False)
    state_exact_formula = state.get("exact_formula", "")

    state_c_min = state.get("c_min", None)
    state_c_max = state.get("c_max", None)
    state_h_min = state.get("h_min", None)
    state_h_max = state.get("h_max", None)
    state_n_min = state.get("n_min", None)
    state_n_max = state.get("n_max", None)
    state_o_min = state.get("o_min", None)
    state_o_max = state.get("o_max", None)
    state_p_min = state.get("p_min", None)
    state_p_max = state.get("p_max", None)
    state_s_min = state.get("s_min", None)
    state_s_max = state.get("s_max", None)

    state_f_state = state.get("f_state", "allowed")
    state_cl_state = state.get("cl_state", "allowed")
    state_br_state = state.get("br_state", "allowed")
    state_i_state = state.get("i_state", "allowed")


    return (
        state,
        state_auto_run,
        state_taxon,
        state_smiles,
        state_smiles_search_type,
        state_smiles_threshold,
        state_mass_filter,
        state_mass_min,
        state_mass_max,
        state_year_filter,
        state_year_start,
        state_year_end,
        state_formula_filter,
        state_exact_formula,
        state_c_min,
        state_c_max,
        state_h_min,
        state_h_max,
        state_n_min,
        state_n_max,
        state_o_min,
        state_o_max,
        state_p_min,
        state_p_max,
        state_s_min,
        state_s_max,
        state_f_state,
        state_cl_state,
        state_br_state,
        state_i_state,
    )


@app.cell
def header(mo):
    """Display application header."""
    mo.md("""
    # 🌐 LOTUS Wikidata Explorer
    """)


@app.cell
def info_callout(mo):
    """Display informational callout."""
    mo.callout(
        mo.md("""
        **Work in progress** - May not work in all deployments.  
        **Recommended:** `uvx marimo run https://raw.githubusercontent.com/Adafede/marimo/refs/heads/main/apps/lotus_wikidata_explorer.py`
        """),
        kind="info",
    )


@app.cell
def help_section(mo, CONFIG):
    """Display help and API documentation."""
    # URL Query API section (left)
    url_api_section = mo.accordion(
        {
            "🔗 URL Query API": mo.md("""
            Query via URL parameters. Add to notebook URL to auto-execute searches.

            **Key Parameters:**
            - `taxon` - Name, QID, or "*" for all
            - `smiles` - Chemical structure
            - `smiles_search_type` - "substructure" or "similarity"
            - `smiles_threshold` - 0.0-1.0 (for similarity)
            - `mass_filter=true`, `mass_min`, `mass_max`
            - `year_filter=true`, `year_start`, `year_end`
            - `formula_filter=true`, `exact_formula`
            - Element ranges: `c_min`, `c_max`, `h_min`, `h_max`, etc.
            - Halogen states: `f_state`, `cl_state`, `br_state`, `i_state`

            **Examples:**
            ```
            ?taxon=Salix&smiles=CC(=O)Oc1ccccc1C(=O)O
            ?smiles=c1ccccc1&smiles_search_type=similarity&smiles_threshold=0.9
            ?taxon=*&mass_filter=true&mass_min=300&mass_max=500
            ```
            """)
        }
    )

    # Help & Documentation section (right)
    help_section = mo.accordion(
        {
            "❓ Help": mo.md("""
            **Quick Start:** Enter taxon (or "*") → Add SMILES (optional) → Apply filters → Search

            **Search Modes:**
            - **Taxon only**: All compounds in that group
            - **SMILES only**: Find structures everywhere
            - **Both**: Find structures in specific taxon

            **Chemical Search** (SACHEM/IDSM):
            - **Substructure**: Find compounds containing your structure
            - **Similarity**: Find similar compounds (Tanimoto 0.0-1.0, default 0.8)

            **Filters:** Mass (Da), Year, Formula (exact or element ranges + halogen control)

            **Export:** CSV, JSON, RDF/Turtle. Auto-compress >8MB.
            """)
        }
    )

    # Display side by side
    mo.hstack([url_api_section, help_section], gap=2, widths="equal")


@app.cell
def url_params_display(mo, format_url_params_display):
    """Display URL parameters if present."""
    _url_params = mo.query_params()
    _params_text = format_url_params_display(_url_params)

    if _params_text:
        mo.callout(
            mo.md(f"**URL Query Detected** - Auto-executing with:\n\n{_params_text}"),
            kind="info",
        )
    else:
        None


@app.cell
def create_search_inputs(
    mo,
    CONFIG,
    state_taxon,
    state_smiles,
    state_smiles_search_type,
    state_smiles_threshold,
    state_mass_filter,
    state_mass_min,
    state_mass_max,
    state_year_filter,
    state_year_start,
    state_year_end,
    state_formula_filter,
    state_exact_formula,
    state_c_min,
    state_c_max,
    state_h_min,
    state_h_max,
    state_n_min,
    state_n_max,
    state_o_min,
    state_o_max,
    state_p_min,
    state_p_max,
    state_s_min,
    state_s_max,
    state_f_state,
    state_cl_state,
    state_br_state,
    state_i_state,
    datetime,
):
    """Create all search input widgets."""
    # Taxon input
    taxon_input = mo.ui.text(
        value=state_taxon,
        label="🔬 Taxon Name or Wikidata QID - Optional",
        placeholder="e.g., Artemisia annua, Cinchona, Q157115, or * for all taxa",
        full_width=True,
    )

    # SMILES input
    smiles_input = mo.ui.text(
        value=state_smiles,
        label="🧪 Chemical Structure (SMILES) - Optional",
        placeholder="e.g., c1ccccc1 (benzene), CC(=O)Oc1ccccc1C(=O)O (aspirin)",
        full_width=True,
    )

    smiles_search_type = mo.ui.dropdown(
        options=["substructure", "similarity"],
        value=state_smiles_search_type,
        label="🔍 Chemical Search Type",
        full_width=True,
    )

    smiles_threshold = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=state_smiles_threshold,
        label="🎯 Similarity Threshold",
        full_width=True,
    )

    # Mass filters
    mass_filter = mo.ui.checkbox(label="⚖️ Filter by mass", value=state_mass_filter)
    mass_min = mo.ui.number(
        value=state_mass_min,
        start=0,
        stop=CONFIG["mass_ui_max"],
        step=0.001,
        label="Min mass (Da)",
        full_width=True,
    )
    mass_max = mo.ui.number(
        value=state_mass_max,
        start=0,
        stop=CONFIG["mass_ui_max"],
        step=0.001,
        label="Max mass (Da)",
        full_width=True,
    )

    # Year filters
    current_year = datetime.now().year
    year_filter = mo.ui.checkbox(
        label="🗓️ Filter by publication year", value=state_year_filter
    )
    year_start = mo.ui.number(
        value=state_year_start,
        start=CONFIG["year_range_start"],
        stop=current_year,
        label="Start year",
        full_width=True,
    )
    year_end = mo.ui.number(
        value=state_year_end,
        start=CONFIG["year_range_start"],
        stop=current_year,
        label="End year",
        full_width=True,
    )

    # Formula filters
    formula_filter = mo.ui.checkbox(
        label="⚛️ Filter by molecular formula", value=state_formula_filter
    )
    exact_formula = mo.ui.text(
        value=state_exact_formula,
        label="Exact formula (e.g., C15H10O5)",
        placeholder="Leave empty to use element ranges",
        full_width=True,
    )

    # Element range inputs
    c_min = mo.ui.number(
        value=state_c_min,
        start=0,
        stop=CONFIG["element_c_max"],
        label="C min",
        full_width=True,
    )
    c_max = mo.ui.number(
        value=state_c_max if state_c_max is not None else CONFIG["element_c_max"],
        start=0,
        stop=CONFIG["element_c_max"],
        label="C max",
        full_width=True,
    )
    h_min = mo.ui.number(
        value=state_h_min,
        start=0,
        stop=CONFIG["element_h_max"],
        label="H min",
        full_width=True,
    )
    h_max = mo.ui.number(
        value=state_h_max if state_h_max is not None else CONFIG["element_h_max"],
        start=0,
        stop=CONFIG["element_h_max"],
        label="H max",
        full_width=True,
    )
    n_min = mo.ui.number(
        value=state_n_min,
        start=0,
        stop=CONFIG["element_n_max"],
        label="N min",
        full_width=True,
    )
    n_max = mo.ui.number(
        value=state_n_max if state_n_max is not None else CONFIG["element_n_max"],
        start=0,
        stop=CONFIG["element_n_max"],
        label="N max",
        full_width=True,
    )
    o_min = mo.ui.number(
        value=state_o_min,
        start=0,
        stop=CONFIG["element_o_max"],
        label="O min",
        full_width=True,
    )
    o_max = mo.ui.number(
        value=state_o_max if state_o_max is not None else CONFIG["element_o_max"],
        start=0,
        stop=CONFIG["element_o_max"],
        label="O max",
        full_width=True,
    )
    p_min = mo.ui.number(
        value=state_p_min,
        start=0,
        stop=CONFIG["element_p_max"],
        label="P min",
        full_width=True,
    )
    p_max = mo.ui.number(
        value=state_p_max if state_p_max is not None else CONFIG["element_p_max"],
        start=0,
        stop=CONFIG["element_p_max"],
        label="P max",
        full_width=True,
    )
    s_min = mo.ui.number(
        value=state_s_min,
        start=0,
        stop=CONFIG["element_s_max"],
        label="S min",
        full_width=True,
    )
    s_max = mo.ui.number(
        value=state_s_max if state_s_max is not None else CONFIG["element_s_max"],
        start=0,
        stop=CONFIG["element_s_max"],
        label="S max",
        full_width=True,
    )

    # Halogen selectors
    halogen_options = ["allowed", "required", "excluded"]
    f_state = mo.ui.dropdown(
        options=halogen_options, value=state_f_state, label="F", full_width=True
    )
    cl_state = mo.ui.dropdown(
        options=halogen_options, value=state_cl_state, label="Cl", full_width=True
    )
    br_state = mo.ui.dropdown(
        options=halogen_options, value=state_br_state, label="Br", full_width=True
    )
    i_state = mo.ui.dropdown(
        options=halogen_options, value=state_i_state, label="I", full_width=True
    )

    # Run button
    run_button = mo.ui.run_button(label="🔍 Search Wikidata")

    return (
        taxon_input,
        smiles_input,
        smiles_search_type,
        smiles_threshold,
        mass_filter,
        mass_min,
        mass_max,
        year_filter,
        year_start,
        year_end,
        formula_filter,
        exact_formula,
        c_min,
        c_max,
        h_min,
        h_max,
        n_min,
        n_max,
        o_min,
        o_max,
        p_min,
        p_max,
        s_min,
        s_max,
        f_state,
        cl_state,
        br_state,
        i_state,
        run_button,
    )


@app.cell
def display_search_ui(
    mo,
    taxon_input,
    smiles_input,
    smiles_search_type,
    smiles_threshold,
    mass_filter,
    mass_min,
    mass_max,
    year_filter,
    year_start,
    year_end,
    formula_filter,
    exact_formula,
    c_min,
    c_max,
    h_min,
    h_max,
    n_min,
    n_max,
    o_min,
    o_max,
    p_min,
    p_max,
    s_min,
    s_max,
    f_state,
    cl_state,
    br_state,
    i_state,
    run_button,
):
    """Build and display the search UI."""
    # Build structure search section
    structure_fields = [smiles_input, smiles_search_type]
    if smiles_search_type.value == "similarity":
        structure_fields.append(smiles_threshold)

    structure_section = mo.vstack(structure_fields)

    # Main search parameters
    main_search = mo.hstack(
        [mo.vstack([taxon_input]), structure_section],
        gap=3,
        widths="equal",
    )

    # Filter buttons
    filter_buttons = mo.hstack(
        [mass_filter, year_filter, formula_filter],
        gap=3,
        justify="start",
    )

    # Build filters UI
    filters_ui = [
        mo.md("## 🔍 Search Parameters"),
        main_search,
        mo.md("### Optional Filters"),
        filter_buttons,
    ]

    # Add conditional filters
    if mass_filter.value:
        filters_ui.append(mo.hstack([mass_min, mass_max], gap=2, widths="equal"))

    if year_filter.value:
        filters_ui.append(mo.hstack([year_start, year_end], gap=2, widths="equal"))

    if formula_filter.value:
        filters_ui.extend(
            [
                exact_formula,
                mo.md("**Element count ranges** (leave blank for no constraint)"),
                mo.hstack([c_min, c_max], gap=2, widths="equal"),
                mo.hstack([h_min, h_max], gap=2, widths="equal"),
                mo.hstack([n_min, n_max], gap=2, widths="equal"),
                mo.hstack([o_min, o_max], gap=2, widths="equal"),
                mo.hstack([p_min, p_max], gap=2, widths="equal"),
                mo.hstack([s_min, s_max], gap=2, widths="equal"),
                mo.md("**Halogen constraints**"),
                mo.hstack(
                    [f_state, cl_state, br_state, i_state], gap=2, widths="equal"
                ),
            ]
        )

    filters_ui.append(run_button)

    mo.vstack(filters_ui)


@app.cell
def execute_search(
    mo,
    time,
    taxon_input,
    smiles_input,
    smiles_search_type,
    smiles_threshold,
    mass_filter,
    mass_min,
    mass_max,
    year_filter,
    year_start,
    year_end,
    formula_filter,
    exact_formula,
    c_min,
    c_max,
    h_min,
    h_max,
    n_min,
    n_max,
    o_min,
    o_max,
    p_min,
    p_max,
    s_min,
    s_max,
    f_state,
    cl_state,
    br_state,
    i_state,
    run_button,
    state_auto_run,
    resolve_taxon_to_qid,
    query_wikidata,
    create_formula_filters,
    create_wikidata_link,
    get_filter_values,
):
    """Execute the search query."""
    # Auto-run if URL parameters were detected, or if run button was clicked
    if not run_button.value and not state_auto_run:
        results_df = None
        qid = None
        taxon_warning = None
    else:
        start_time = time.time()

        # Get input values
        smiles_str = smiles_input.value.strip() if smiles_input.value else ""
        taxon_input_str = taxon_input.value.strip()

        # Determine search mode
        use_smiles = bool(smiles_str)
        use_taxon = bool(taxon_input_str and taxon_input_str != "*")

        if use_smiles and use_taxon:
            spinner_message = (
                f"🔎 Searching for SMILES '{smiles_str[:30]}...' in {taxon_input_str}"
            )
        elif use_smiles:
            spinner_message = f"🔎 Searching for SMILES: {smiles_str[:50]}..."
            qid = None
            taxon_warning = None
        else:
            if taxon_input_str == "*":
                spinner_message = "🔎 Searching all taxa ..."
            else:
                spinner_message = f"🔎 Searching for: {taxon_input_str}"

        with mo.status.spinner(title=spinner_message):
            # Resolve taxon if needed
            if use_taxon or (not use_smiles and taxon_input_str):
                taxon_input_str = taxon_input.value.strip()
                qid, taxon_warning = resolve_taxon_to_qid(taxon_input_str)
                if not qid:
                    mo.stop(
                        True,
                        mo.callout(
                            mo.md(
                                f"**Taxon not found**\n\n"
                                f"Could not find '{taxon_input_str}' in Wikidata.\n\n"
                                f"**Suggestions:**\n"
                                f"- Check spelling (scientific names are case-sensitive)\n"
                                f"- Try a different taxonomic level (e.g., genus instead of species)\n"
                                f"- Use a Wikidata QID directly (e.g., Q157115)"
                            ),
                            kind="warn",
                        ),
                    )

            try:
                # Extract filter values
                y_start, y_end, m_min, m_max = get_filter_values(
                    mass_filter, mass_min, mass_max, year_filter, year_start, year_end
                )

                # Build formula filters
                formula_filt = None
                if formula_filter.value:
                    formula_filt = create_formula_filters(
                        exact_formula=exact_formula.value,
                        c_min=c_min.value,
                        c_max=c_max.value,
                        h_min=h_min.value,
                        h_max=h_max.value,
                        n_min=n_min.value,
                        n_max=n_max.value,
                        o_min=o_min.value,
                        o_max=o_max.value,
                        p_min=p_min.value,
                        p_max=p_max.value,
                        s_min=s_min.value,
                        s_max=s_max.value,
                        f_state=f_state.value,
                        cl_state=cl_state.value,
                        br_state=br_state.value,
                        i_state=i_state.value,
                    )

                # Execute query based on search mode
                if use_smiles and use_taxon:
                    results_df = query_wikidata(
                        qid=qid,
                        mass_min=m_min,
                        mass_max=m_max,
                        formula_filters=formula_filt,
                        smiles=smiles_str,
                        search_mode="combined",
                        smiles_search_type=smiles_search_type.value,
                        smiles_threshold=smiles_threshold.value,
                    )
                elif use_smiles:
                    results_df = query_wikidata(
                        qid="",
                        mass_min=m_min,
                        mass_max=m_max,
                        formula_filters=formula_filt,
                        smiles=smiles_str,
                        search_mode="smiles",
                        smiles_search_type=smiles_search_type.value,
                        smiles_threshold=smiles_threshold.value,
                    )
                else:
                    results_df = query_wikidata(
                        qid, y_start, y_end, m_min, m_max, formula_filt
                    )
            except Exception as e:
                mo.stop(
                    True, mo.callout(mo.md(f"**Query Error:** {str(e)}"), kind="danger")
                )

        elapsed = round(time.time() - start_time, 2)
        mo.md(f"⏱️ Query completed in **{elapsed}s**.")

    return qid, results_df, taxon_warning


@app.cell
def compute_hashes(
    results_df,
    qid,
    taxon_input,
    mass_filter,
    mass_min,
    mass_max,
    year_filter,
    year_start,
    year_end,
    formula_filter,
    exact_formula,
    c_min,
    c_max,
    h_min,
    h_max,
    n_min,
    n_max,
    o_min,
    o_max,
    p_min,
    p_max,
    s_min,
    s_max,
    f_state,
    cl_state,
    br_state,
    i_state,
    smiles_input,
    smiles_search_type,
    smiles_threshold,
    create_dataset_hashes,
    build_filters_dict,
):
    """Compute provenance hashes for the results."""
    if results_df is None or len(results_df) == 0:
        query_hash = None
        result_hash = None
    else:
        # Build filters dict for hashing
        _filters = build_filters_dict(
            smiles_input,
            smiles_search_type,
            smiles_threshold,
            mass_filter,
            mass_min,
            mass_max,
            year_filter,
            year_start,
            year_end,
            formula_filter,
            exact_formula,
            c_min,
            c_max,
            h_min,
            h_max,
            n_min,
            n_max,
            o_min,
            o_max,
            p_min,
            p_max,
            s_min,
            s_max,
            f_state,
            cl_state,
            br_state,
            i_state,
        )

        query_hash, result_hash = create_dataset_hashes(
            qid or "", taxon_input.value, _filters, results_df
        )

    return query_hash, result_hash


@app.cell
def display_results_summary(
    mo,
    results_df,
    qid,
    taxon_input,
    taxon_warning,
    smiles_input,
    smiles_search_type,
    smiles_threshold,
    mass_filter,
    mass_min,
    mass_max,
    year_filter,
    year_start,
    year_end,
    formula_filter,
    exact_formula,
    c_min,
    c_max,
    h_min,
    h_max,
    n_min,
    n_max,
    o_min,
    o_max,
    p_min,
    p_max,
    s_min,
    s_max,
    f_state,
    cl_state,
    br_state,
    i_state,
    query_hash,
    result_hash,
    run_button,
    state_auto_run,
    create_wikidata_link,
    pluralize,
    build_api_url,
):
    """Display results summary and statistics."""
    if (not run_button.value and not state_auto_run) or results_df is None:
        summary_and_downloads = mo.Html("")
    elif len(results_df) == 0:
        # No results found
        parts = []
        if taxon_warning:
            parts.append(mo.callout(taxon_warning, kind="warn"))

        if qid == "*":
            parts.append(
                mo.callout(
                    mo.md(
                        "No natural products found for **all taxa** with the current filters."
                    ),
                    kind="warn",
                )
            )
        else:
            parts.append(
                mo.callout(
                    mo.md(
                        f"No natural products found for **{taxon_input.value}** ({create_wikidata_link(qid)}) with the current filters."
                    ),
                    kind="warn",
                )
            )
        summary_and_downloads = mo.vstack(parts) if len(parts) > 1 else parts[0]
    else:
        # Results found
        n_compounds = results_df.n_unique(subset=["compound"])
        n_taxa = results_df.n_unique(subset=["taxon"])
        n_refs = results_df.n_unique(subset=["reference"])
        n_entries = len(results_df)

        # Results header
        results_header = mo.md("## Results")

        # Taxon info
        if qid == "*":
            taxon_info = "All taxa"
        else:
            taxon_info = f"{taxon_input.value} {create_wikidata_link(qid)}"

        # Add SMILES info if present
        if smiles_input.value and smiles_input.value.strip():
            _smiles_str = smiles_input.value.strip()
            search_type = smiles_search_type.value

            if search_type == "similarity":
                threshold_val = smiles_threshold.value
                smiles_info = f"SMILES: `{_smiles_str}` ({search_type}, threshold: {threshold_val})"
            else:
                smiles_info = f"SMILES: `{_smiles_str[:50]}...` ({search_type})"

            combined_info = f"{taxon_info} • {smiles_info}"
        else:
            combined_info = taxon_info

        # Search info
        search_info_display = mo.md(f"**{combined_info}**")

        # Provenance hash
        hash_info = mo.md(f"*Hashes:* Query: `{query_hash}` • Results: `{result_hash}`")

        # Stats cards
        stats_cards = mo.hstack(
            [
                mo.stat(
                    value=f"{n_compounds:,}",
                    label=f"🧪 {pluralize('Compound', n_compounds)}",
                    bordered=True,
                ),
                mo.stat(
                    value=f"{n_taxa:,}",
                    label=f"🌿 {pluralize('Taxon', n_taxa)}",
                    bordered=True,
                ),
                mo.stat(
                    value=f"{n_refs:,}",
                    label=f"📚 {pluralize('Reference', n_refs)}",
                    bordered=True,
                ),
                mo.stat(
                    value=f"{n_entries:,}",
                    label=f"📊 {pluralize('Entry', n_entries)}",
                    bordered=True,
                ),
            ],
            gap=2,
        )

        # Build API URL for sharing
        api_url = build_api_url(
            taxon=taxon_input.value,
            smiles=smiles_input.value,
            smiles_search_type=smiles_search_type.value,
            smiles_threshold=smiles_threshold.value,
            mass_filter=mass_filter.value,
            mass_min=mass_min.value,
            mass_max=mass_max.value,
            year_filter=year_filter.value,
            year_start=year_start.value,
            year_end=year_end.value,
            formula_filter=formula_filter.value,
            exact_formula=exact_formula.value,
            c_min=c_min.value,
            c_max=c_max.value,
            h_min=h_min.value,
            h_max=h_max.value,
            n_min=n_min.value,
            n_max=n_max.value,
            o_min=o_min.value,
            o_max=o_max.value,
            p_min=p_min.value,
            p_max=p_max.value,
            s_min=s_min.value,
            s_max=s_max.value,
            f_state=f_state.value,
            cl_state=cl_state.value,
            br_state=br_state.value,
            i_state=i_state.value,
        )

        # Display shareable URL if parameters exist
        if api_url:
            url_display = mo.md(
                f"""
                **🔗 Shareable URL**

                Copy and append this to your notebook URL to share this exact search:
                ```
                {api_url}
                ```
                """
            )
            api_url_section = mo.accordion(
                {"🔁 Share this search": url_display},
                multiple=False,
            )
        else:
            api_url_section = None

        # Combine all parts
        summary_parts = [results_header, search_info_display, hash_info]
        if api_url_section:
            summary_parts.append(api_url_section)
        if taxon_warning:
            summary_parts.append(mo.callout(taxon_warning, kind="warn"))
        summary_parts.append(stats_cards)

        summary_and_downloads = mo.vstack(summary_parts)

    return summary_and_downloads


@app.cell
def render_summary(summary_and_downloads):
    """Render the summary section."""
    summary_and_downloads


@app.cell
def display_results_table(
    mo,
    results_df,
    prepare_export_dataframe,
    CONFIG,
    run_button,
    state_auto_run,
    create_link,
    create_wikidata_link,
    create_structure_image_url,
):
    """Display results table with formatted data."""
    if (
        (not run_button.value and not state_auto_run)
        or results_df is None
        or len(results_df) == 0
    ):
        results_table = mo.Html("")
    else:
        # Limit rows for better table performance
        table_row_limit = CONFIG["table_row_limit"]

        if len(results_df) > table_row_limit:
            # Only prepare the limited rows for display
            display_df = prepare_export_dataframe(
                results_df.head(table_row_limit), include_rdf_ref=False
            )
            warning_msg = mo.callout(
                mo.md(
                    f"⚠️ Showing first {table_row_limit:,} of {len(results_df):,} rows. "
                    f"Export full dataset using download buttons below."
                ),
                kind="warn",
            )
        else:
            # Prepare all rows
            display_df = prepare_export_dataframe(results_df, include_rdf_ref=False)
            warning_msg = None

        # Create table with inline structure images
        table = mo.ui.table(
            display_df,
            selection=None,
            page_size=CONFIG["page_size_default"],
            format_mapping={
                "compound_qid": lambda qid: create_wikidata_link(
                    qid, color=CONFIG["color_wikidata_red"]
                ),
                "taxon_qid": lambda qid: create_wikidata_link(
                    qid, color=CONFIG["color_wikidata_green"]
                ),
                "reference_qid": lambda qid: create_wikidata_link(
                    qid, color=CONFIG["color_wikidata_blue"]
                ),
                "reference_doi": lambda doi: create_link(f"https://doi.org/{doi}", doi)
                if doi
                else mo.Html(""),
                "compound_smiles": lambda smiles: mo.Html(
                    f'<img src="{create_structure_image_url(smiles)}" width="120" height="120" />'
                )
                if smiles
                else mo.Html(""),
            },
        )

        # Combine warning and table
        if warning_msg:
            results_table = mo.vstack([warning_msg, table])
        else:
            results_table = table

    return results_table


@app.cell
def render_table(results_table):
    """Render the results table."""
    results_table


@app.cell
def create_downloads(
    mo,
    results_df,
    qid,
    taxon_input,
    query_hash,
    result_hash,
    mass_filter,
    mass_min,
    mass_max,
    year_filter,
    year_start,
    year_end,
    formula_filter,
    exact_formula,
    c_min,
    c_max,
    h_min,
    h_max,
    n_min,
    n_max,
    o_min,
    o_max,
    p_min,
    p_max,
    s_min,
    s_max,
    f_state,
    cl_state,
    br_state,
    i_state,
    smiles_input,
    smiles_search_type,
    smiles_threshold,
    run_button,
    state_auto_run,
    prepare_export_dataframe,
    create_export_metadata,
    export_to_rdf_turtle,
    generate_filename,
    compress_if_large,
    create_citation_text,
):
    """Create download buttons for exports."""
    if (
        (not run_button.value and not state_auto_run)
        or results_df is None
        or len(results_df) == 0
    ):
        download_ui = mo.Html("")
    else:
        # Build filters dict
        _filters = build_filters_dict(
            smiles_input,
            smiles_search_type,
            smiles_threshold,
            mass_filter,
            mass_min,
            mass_max,
            year_filter,
            year_start,
            year_end,
            formula_filter,
            exact_formula,
            c_min,
            c_max,
            h_min,
            h_max,
            n_min,
            n_max,
            o_min,
            o_max,
            p_min,
            p_max,
            s_min,
            s_max,
            f_state,
            cl_state,
            br_state,
            i_state,
        )

        # Prepare export dataframe for all export formats
        _export_df = prepare_export_dataframe(results_df, include_rdf_ref=True)

        # Generate metadata
        metadata = create_export_metadata(
            _export_df, taxon_input.value, qid or "", _filters, query_hash, result_hash
        )

        # Generate export data on-demand when button is clicked
        def _generate_csv():
            csv_data = _export_df.write_csv()
            csv_bytes, compressed = compress_if_large(csv_data.encode("utf-8"))
            return csv_bytes

        def _generate_json():
            json_data = _export_df.write_json()
            json_bytes, compressed = compress_if_large(json_data.encode("utf-8"))
            return json_bytes

        def _generate_rdf():
            rdf_data = export_to_rdf_turtle(
                _export_df, taxon_input.value, qid or "", _filters
            )
            rdf_bytes, compressed = compress_if_large(rdf_data.encode("utf-8"))
            return rdf_bytes

        # Generate filenames and create buttons
        csv_filename = generate_filename(taxon_input.value, "csv", filters=_filters)
        json_filename = generate_filename(taxon_input.value, "json", filters=_filters)
        rdf_filename = generate_filename(taxon_input.value, "ttl", filters=_filters)

        # Note: Check compression after generation for accurate filename
        csv_button = mo.download(
            data=_generate_csv,
            filename=csv_filename,
            label="📥 Download CSV",
        )

        json_button = mo.download(
            data=_generate_json,
            filename=json_filename,
            label="📥 Download JSON",
        )

        rdf_button = mo.download(
            data=_generate_rdf,
            filename=rdf_filename,
            label="📥 Download RDF/Turtle",
        )

        # Metadata export (already lightweight)
        import json as json_lib

        metadata_json = json_lib.dumps(metadata, indent=2)
        metadata_button = mo.download(
            metadata_json.encode("utf-8"),
            filename=generate_filename(
                taxon_input.value, "metadata.json", filters=_filters
            ),
            label="📥 Download Metadata",
        )

        # Citation info
        citation_md = create_citation_text(taxon_input.value)

        download_ui = mo.vstack(
            [
                mo.md("### 📥 Export Options"),
                mo.hstack(
                    [csv_button, json_button, rdf_button, metadata_button], gap=2
                ),
                mo.accordion({"📖 Citation Information": mo.md(citation_md)}),
            ]
        )

    return download_ui


@app.cell
def display_downloads(download_ui):
    """Render download section."""
    download_ui




@app.cell
def footer(mo):
    """Display footer with attribution and license information."""
    mo.md("""
    ---
    **Data:** <a href="https://www.wikidata.org/wiki/Q104225190" style="color:#990000;">LOTUS Initiative</a> & <a href="https://www.wikidata.org/" style="color:#990000;">Wikidata</a>  |
    **Code:** <a href="https://github.com/cdk/depict" style="color:#339966;">CDK Depict</a> & <a href="https://github.com/Adafede/marimo/blob/main/apps/lotus_wikidata_explorer_refactored.py" style="color:#339966;">lotus_wikidata_explorer_refactored.py</a>  |
    **License:** <a href="https://creativecommons.org/publicdomain/zero/1.0/" style="color:#006699;">CC0 1.0</a> for data & <a href="https://www.gnu.org/licenses/agpl-3.0.html" style="color:#006699;">AGPL-3.0</a> for code
    """)


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "export":
        # CLI mode - extract and reuse app.setup functions
        import argparse
        import gzip
        import hashlib
        import io
        import polars as pl
        from pathlib import Path

        # Parse CLI arguments
        parser = argparse.ArgumentParser(description="Export LOTUS data")
        parser.add_argument("export")
        parser.add_argument("--taxon", help="Taxon name or QID")
        parser.add_argument("--output", "-o", help="Output file")
        parser.add_argument(
            "--format", "-f", choices=["csv", "json", "ttl"], default="csv"
        )
        parser.add_argument("--year-start", type=int, help="Minimum publication year")
        parser.add_argument("--year-end", type=int, help="Maximum publication year")
        parser.add_argument("--mass-min", type=float, help="Minimum molecular mass")
        parser.add_argument("--mass-max", type=float, help="Maximum molecular mass")

        # Molecular formula filters
        parser.add_argument(
            "--formula", help="Exact molecular formula (e.g., C15H10O5)"
        )
        parser.add_argument("--c-min", type=int, help="Minimum carbon atoms")
        parser.add_argument("--c-max", type=int, help="Maximum carbon atoms")
        parser.add_argument("--h-min", type=int, help="Minimum hydrogen atoms")
        parser.add_argument("--h-max", type=int, help="Maximum hydrogen atoms")
        parser.add_argument("--n-min", type=int, help="Minimum nitrogen atoms")
        parser.add_argument("--n-max", type=int, help="Maximum nitrogen atoms")
        parser.add_argument("--o-min", type=int, help="Minimum oxygen atoms")
        parser.add_argument("--o-max", type=int, help="Maximum oxygen atoms")

        parser.add_argument(
            "--smiles", help="SMILES string for chemical structure search"
        )
        parser.add_argument(
            "--smiles-search-type",
            choices=["exact", "substructure", "similarity"],
            help="Type of SMILES search (exact, substructure, or similarity)",
        )
        parser.add_argument(
            "--smiles-threshold",
            type=float,
            default=0.8,
            help="Similarity threshold (0.0-1.0, default: 0.8)",
        )
        parser.add_argument(
            "--compress", action="store_true", help="Compress output with gzip"
        )
        parser.add_argument(
            "--show-metadata",
            action="store_true",
            help="Display FAIR-compliant dataset metadata and exit (no data export)",
        )
        parser.add_argument(
            "--export-metadata",
            action="store_true",
            help="Export metadata as JSON file alongside data (creates <output>.metadata.json)",
        )
        parser.add_argument(
            "--verbose", "-v", action="store_true", help="Verbose output"
        )
        args = parser.parse_args()

        try:
            with open(__file__, "r", encoding="utf-8") as f:
                file_content = f.read()

            # Strategy: Extract app.setup block AND all @app.function decorated functions
            # This ensures we get imports, constants, and all the actual functions
            setup_start = file_content.find("with app.setup:")
            if setup_start == -1:
                raise RuntimeError("Could not find app.setup block")

            setup_start += len("with app.setup:\n")
            next_app_cell = file_content.find("\n@app.", setup_start)

            if next_app_cell == -1:
                raise RuntimeError("Could not find end of app.setup block")

            setup_block = file_content[setup_start:next_app_cell]

            # Dedent app.setup (remove 4-space indentation)
            setup_lines = []
            for line in setup_block.split("\n"):
                if line.startswith("    "):
                    setup_lines.append(line[4:])
                elif line.strip() == "":
                    setup_lines.append("")
                else:
                    break

            # Extract all @app.function blocks (they contain the actual functions)
            import re

            function_pattern = r"@app\.function\s*\ndef\s+(\w+)"
            function_blocks = []

            # Find all @app.function decorated functions
            for match in re.finditer(function_pattern, file_content):
                func_name = match.group(1)
                func_start = match.start()

                # Find where this function ends (next @app. or if __name__)
                next_decorator = file_content.find("\n@app.", func_start + 1)
                next_main = file_content.find("\nif __name__", func_start)

                if next_decorator != -1:
                    func_end = next_decorator
                elif next_main != -1:
                    func_end = next_main
                else:
                    func_end = len(file_content)

                # Extract function definition (skip the @app.function decorator)
                func_def_start = file_content.find("def " + func_name, func_start)
                func_block = file_content[func_def_start:func_end].strip()
                function_blocks.append(func_block)

            # Combine setup and functions into executable code
            combined_code = (
                "\n".join(setup_lines) + "\n\n" + "\n\n".join(function_blocks)
            )

            # Execute in isolated namespace
            namespace = {}
            exec(combined_code, namespace)

            if args.taxon is None:
                args.taxon = "*"

            if args.verbose:
                print(f"Querying LOTUS data for: {args.taxon}", file=sys.stderr)

            # Resolve taxon using the REAL function
            qid, warning = resolve_taxon_to_qid(args.taxon)
            if not qid:
                print(f"❌ Taxon not found: {args.taxon}", file=sys.stderr)
                sys.exit(1)

            # Determine search mode based on whether SMILES is provided
            search_mode = "taxon"
            if args.smiles:
                search_mode = "combined" if qid and qid != "*" else "smiles"

            # Improved verbose logging
            if args.verbose:
                print(f"🔍 Search Configuration:", file=sys.stderr)
                print(f"   Mode: {search_mode}", file=sys.stderr)
                print(f"   Taxon: {args.taxon} (QID: {qid})", file=sys.stderr)

                if args.smiles:
                    print(f"   SMILES: {args.smiles}", file=sys.stderr)
                    print(
                        f"   Search Type: {args.smiles_search_type or 'substructure'}",
                        file=sys.stderr,
                    )
                    if args.smiles_search_type == "similarity":
                        print(
                            f"   Similarity Threshold: {args.smiles_threshold}",
                            file=sys.stderr,
                        )

                filters_applied = []
                if args.year_start:
                    filters_applied.append(f"year ≥ {args.year_start}")
                if args.year_end:
                    filters_applied.append(f"year ≤ {args.year_end}")
                if args.mass_min:
                    filters_applied.append(f"mass ≥ {args.mass_min}")
                if args.mass_max:
                    filters_applied.append(f"mass ≤ {args.mass_max}")
                if args.formula:
                    filters_applied.append(f"formula = {args.formula}")
                if any(
                    [
                        args.c_min,
                        args.c_max,
                        args.h_min,
                        args.h_max,
                        args.n_min,
                        args.n_max,
                        args.o_min,
                        args.o_max,
                    ]
                ):
                    filters_applied.append("element ranges")

                if filters_applied:
                    print(f"   Filters: {', '.join(filters_applied)}", file=sys.stderr)

                print(file=sys.stderr)  # Empty line for readability

            # Build formula filters if any formula arguments provided
            formula_filt = None
            if any(
                [
                    args.formula,
                    args.c_min,
                    args.c_max,
                    args.h_min,
                    args.h_max,
                    args.n_min,
                    args.n_max,
                    args.o_min,
                    args.o_max,
                ]
            ):
                formula_filt = create_formula_filters(
                    exact_formula=args.formula or "",
                    c_min=args.c_min or 0,
                    c_max=args.c_max or CONFIG["element_c_max"],
                    h_min=args.h_min or 0,
                    h_max=args.h_max or CONFIG["element_h_max"],
                    n_min=args.n_min or 0,
                    n_max=args.n_max or CONFIG["element_n_max"],
                    o_min=args.o_min or 0,
                    o_max=args.o_max or CONFIG["element_o_max"],
                    p_min=0,
                    p_max=CONFIG["element_p_max"],
                    s_min=0,
                    s_max=CONFIG["element_s_max"],
                    f_state="allowed",
                    cl_state="allowed",
                    br_state="allowed",
                    i_state="allowed",
                )

            # Query using the REAL function with all arguments
            df = query_wikidata(
                qid=qid,
                year_start=args.year_start,
                year_end=args.year_end,
                mass_min=args.mass_min,
                mass_max=args.mass_max,
                formula_filters=formula_filt,
                smiles=args.smiles,
                search_mode=search_mode,
                smiles_search_type=args.smiles_search_type or "substructure",
                smiles_threshold=args.smiles_threshold,
            )

            if df.is_empty():
                print(f"❌ No data found", file=sys.stderr)
                sys.exit(1)

            if args.verbose:
                print(f"✅ Query Results:", file=sys.stderr)
                print(f"   Total entries: {len(df):,}", file=sys.stderr)

                # Show unique counts if columns exist
                if "compound" in df.columns:
                    unique_compounds = df.select(pl.col("compound")).n_unique()
                    print(f"   Unique compounds: {unique_compounds:,}", file=sys.stderr)
                if "taxon" in df.columns:
                    unique_taxa = df.select(pl.col("taxon")).n_unique()
                    print(f"   Unique taxa: {unique_taxa:,}", file=sys.stderr)
                if "reference" in df.columns:
                    unique_refs = df.select(pl.col("reference")).n_unique()
                    print(f"   Unique references: {unique_refs:,}", file=sys.stderr)
                print(file=sys.stderr)  # Empty line for readability

            # Build filters dict in the same format as the UI (needed for metadata and provenance)
            import json

            filters = {}

            # Year filter
            if args.year_start or args.year_end:
                filters["publication_year"] = {}
                if args.year_start:
                    filters["publication_year"]["min"] = args.year_start
                if args.year_end:
                    filters["publication_year"]["max"] = args.year_end

            # Mass filter
            if args.mass_min or args.mass_max:
                filters["molecular_mass"] = {}
                if args.mass_min:
                    filters["molecular_mass"]["min"] = args.mass_min
                if args.mass_max:
                    filters["molecular_mass"]["max"] = args.mass_max

            # Chemical structure filter
            if args.smiles:
                filters["chemical_structure"] = {
                    "smiles": args.smiles,
                    "search_type": args.smiles_search_type or "substructure",
                }
                if args.smiles_search_type == "similarity":
                    filters["chemical_structure"]["similarity_threshold"] = (
                        args.smiles_threshold
                    )

            # Molecular formula filter
            if formula_filt and formula_filt.is_active():
                filters["molecular_formula"] = serialize_formula_filters(formula_filt)

            # Compute hashes for provenance (before showing metadata or exporting)
            # Query hash - based on search parameters (what was asked)
            query_components = [qid or "", args.taxon or ""]
            if filters:
                query_components.append(json.dumps(filters, sort_keys=True))
            query_hash = hashlib.sha256(
                "|".join(query_components).encode("utf-8")
            ).hexdigest()

            # Result hash - based on actual compound identifiers (what was found)
            compound_qids = sorted(
                [
                    extract_qid(row["compound"])
                    for row in df.iter_rows(named=True)
                    if row.get("compound")
                ]
            )
            result_hash = hashlib.sha256(
                "|".join(compound_qids).encode("utf-8")
            ).hexdigest()

            # Show metadata mode - use the REAL create_export_metadata function
            if args.show_metadata:
                # Use the REAL metadata function from app.setup with provenance hashes
                metadata = create_export_metadata(
                    df,
                    args.taxon,
                    qid,
                    filters if filters else None,
                    query_hash=query_hash,
                    result_hash=result_hash,
                )
                print(json.dumps(metadata, indent=2))

            # Export data using REAL functions
            if args.format == "csv":
                # CSV export: exclude ref column (statement included for transparency)
                export_df = prepare_export_dataframe(
                    df,
                    include_rdf_ref=False,
                )
                data = export_df.write_csv().encode("utf-8")
            elif args.format == "json":
                # JSON export: exclude ref column (statement included for transparency)
                export_df = prepare_export_dataframe(
                    df,
                    include_rdf_ref=False,
                )
                data = export_df.write_json().encode("utf-8")
            elif args.format == "ttl":
                # RDF export: include ref column for full provenance
                export_df = prepare_export_dataframe(df, include_rdf_ref=True)
                data = export_to_rdf_turtle(
                    export_df, args.taxon, qid, filters if filters else None
                ).encode("utf-8")
            else:
                print(f"❌ Unknown format: {args.format}", file=sys.stderr)
                sys.exit(1)

            # Compress if requested
            if args.compress:
                buffer = io.BytesIO()
                with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
                    gz.write(data)
                data = buffer.getvalue()

            # Write output
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(data)
                if args.verbose:
                    print(
                        f"✓ Exported {len(data):,} bytes to: {output_path}",
                        file=sys.stderr,
                    )

                # Export metadata if requested
                if args.export_metadata:
                    metadata = create_export_metadata(
                        df,
                        args.taxon,
                        qid,
                        filters if filters else None,
                        query_hash=query_hash,
                        result_hash=result_hash,
                    )
                    metadata_path = output_path.with_suffix(
                        output_path.suffix + ".metadata.json"
                    )
                    metadata_path.write_text(json.dumps(metadata, indent=2))
                    if args.verbose:
                        print(
                            f"✓ Exported metadata to: {metadata_path}",
                            file=sys.stderr,
                        )
            else:
                sys.stdout.buffer.write(data)

        except Exception as e:
            print(f"❌ Error: {e}", file=sys.stderr)
            if args.verbose:
                import traceback

                traceback.print_exc()
            sys.exit(1)
    else:
        app.run()


if __name__ == "__main__":
    main()
