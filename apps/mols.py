# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "marimo",
#     "rdkit==2025.9.3",
# ]
# ///

import marimo

NBDIR = marimo.notebook_dir()

__generated_with = "0.19.4"
app = marimo.App(
    width="medium",
    app_title="Automated substructure depiction and verification",
    css_file=NBDIR / "public/custom.css",
    html_head_file=NBDIR / "public/head.html",
)

with app.setup:
    import marimo as mo
    from collections import defaultdict
    from itertools import cycle
    from dataclasses import dataclass

    @dataclass
    class MockInput:
        value: str = ""

    # Toggle this flag for local vs remote development
    _USE_LOCAL = False  # Set to True for local development
    if _USE_LOCAL:
        import sys

        # Add your local module directory to the path
        # Adjust this path to where your "modules" folder is located locally
        sys.path.insert(0, ".")
    from modules.text.strings.parse_labeled_lines import parse_labeled_lines
    from modules.utils.colors.hex_to_rgb_float import hex_to_rgb_float

    try:
        from modules.chem.rdkit.smarts.find_mcs import find_mcs
        from modules.chem.rdkit.smarts.parse import parse as parse_smarts
        from modules.chem.rdkit.smiles.parse_many import (
            parse_many as parse_smiles_list,
        )
        from modules.chem.rdkit.depict.with_highlights import (
            with_highlights as depict_with_highlights,
        )

        message = mo.md("[+] Your environment supports **RDKit**, all good!")
        rdkit_available = True
    except ImportError:
        message = mo.md(
            """
            [!] **RDKit not available in this environment**.
            To run this script locally:

            ```
            uvx marimo run https://adafede.github.io/marimo/apps/mols.py
            ```

            If using Docker, toggle **App View** (bottom right or `cmd + .`).
            """,
        )
        rdkit_available = False
        parse_smarts = None
        find_mcs = None
        parse_smiles_list = None
        depict_with_highlights = None


@app.cell
def message_md():
    message
    return


@app.cell
def stop_rdkit():
    mo.stop(predicate=not rdkit_available)
    return


@app.cell
def input_smiles():
    if rdkit_available:
        smi_input = mo.ui.text_area(
            label="## Enter SMILES (one per line)",
            placeholder="e.g., CCO Ethanol",
            value="CC(=O)O acetic acid\nCCN(CC)CCN N,N-Diethylethylenediamine\nC[C@@H]1C=C(OC)C([C@@]2(C)[C@H]1C[C@@H]3[C@]4(C)[C@@H]2C(C(OC)=C(C)[C@@H]4CC(O3)=O)=O)=O quassin",
            full_width=True,
        )
    else:
        smi_input = MockInput()
    smi_input
    return (smi_input,)


@app.cell
def py_find_mcs(smi_input):
    if rdkit_available:
        smiles_list = parse_labeled_lines(smi_input.value)
        valid_mols = parse_smiles_list(smiles_list=smiles_list)
        mcs_smarts, mcs_error = find_mcs(mols=valid_mols)

        if mcs_smarts:
            mcs = mo.md(
                "### Automatically Detected Maximum Common Substructure (MCS) SMARTS\n\n"
                "The SMARTS pattern below was generated automatically. It may not always be chemically meaningful or appropriate for your use case, so please review it carefully.\n\n"
                "You can paste it below as a starting point:\n"
                f"```smarts\n{mcs_smarts}\n```",
            )
        elif mcs_error:
            mcs = mo.md(f"[!] {mcs_error}")
        else:
            mcs = mo.md("[i] No MCS SMARTS generated.")
    else:
        mcs = None
    mcs
    return


@app.cell
def input_smarts():
    if rdkit_available:
        smarts_input = mo.ui.text_area(
            label="## Enter SMARTS patterns (one per line)",
            placeholder="e.g., [OH] Hydroxyl",
            value="[#7] nitrogen\n[OH] hydroxyl\n[$(C);!$(C(~C)~C)]~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)(~C)~C)]1~[$(C);!$(C(~C)(~C)(~C)~C)](~[$(C);!$(C(~C)~C)])~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)(~C)~C)]2C3([$(C);!$(C(~C)~C)])~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)(~C)~C)](~[$(C);!$(C(~C)~C)])~[$(C);!$(C(~C)(~C)(~C)~C)]3~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)~C)]C2([$(C);!$(C(~C)~C)])1 picrasane",
            full_width=True,
        )
    else:
        smarts_input = MockInput()
    smarts_input
    return (smarts_input,)


@app.cell
def input_toggle(smarts_input):
    smarts_list = parse_labeled_lines(smarts_input.value)
    toggles = {
        smarts: mo.ui.switch(value=True, label=name) for name, smarts in smarts_list
    }

    mo.md("## Toggle SMARTS Highlights")
    for switch in toggles.values():
        switch
    return (toggles,)


@app.cell
def button_submit():
    if rdkit_available:
        submit_button = mo.ui.button(label="Render Molecules")
    else:
        submit_button = MockInput()
    submit_button
    return (submit_button,)


@app.cell
def py_generate_html(smarts_input, smi_input, submit_button, toggles):
    if not rdkit_available:
        html = ""

    _ = submit_button.value  # Trigger re-render

    highlight_palette = [
        "#77aadd",
        "#ee8866",
        "#eedd88",
        "#ffaabb",
        "#99ddff",
        "#44bb99",
        "#bbcc33",
        "#aaaa00",
        "#dddddd",
    ]
    color_cycle = cycle(highlight_palette)

    smiles = parse_labeled_lines(smi_input.value)
    raw_smarts = parse_labeled_lines(smarts_input.value)

    active_smarts = [
        (name, smarts) for name, smarts in raw_smarts if toggles[smarts].value
    ]

    parsed_smarts = [
        (name, smarts, mol, hex_to_rgb_float(hex_color=color))
        for (name, smarts), color in zip(active_smarts, color_cycle)
        if (mol := parse_smarts(smarts=smarts)) is not None
    ]

    match_counter = defaultdict(int)

    if not smiles:
        html = (
            "<p style='color:orange;'>[!] Please enter at least one SMILES string.</p>"
        )
    else:
        rendered = [
            depict_with_highlights(
                name=name,
                smiles=smi,
                smarts_entries=parsed_smarts,
                match_counter=match_counter,
            )
            for name, smi in smiles
        ]

        total_mols = len(smiles)

        summary_items = [
            f"<div style='display:flex; justify-content:space-between; align-items:center; "
            f"padding:6px 12px; border-bottom:1px solid #eee;'>"
            f"<span style='font-size:0.9em; font-weight:500;'>{name}</span>"
            f"<span style='background:{highlight_palette[i % len(highlight_palette)]};"
            f" padding:2px 8px; border-radius:12px; font-size:0.8em;'>"
            f"{match_counter[name]} / {total_mols} mol{'s' if total_mols != 1 else ''}</span>"
            f"</div>"
            for i, (name, _) in enumerate(active_smarts)
        ]

        summary_html = (
            "<div style='margin: 16px auto; padding:12px 16px; max-width:800px; "
            "border: 1px solid #ddd; border-radius: 8px; "
            "box-shadow: 1px 1px 5px rgba(0,0,0,0.05); font-family: sans-serif;'>"
            "<div style='font-weight:bold; font-size:1.1em; margin-bottom:8px;'>SMARTS Match Summary</div>"
            + "".join(summary_items)
            + "</div>"
        )

        html = (
            summary_html
            + "<div style='display:flex; flex-wrap:wrap; justify-content:center;'>"
            + "".join(rendered)
            + "</div>"
        )
    return (html,)


@app.cell
def html_display(html):
    mo.Html(html)
    return


if __name__ == "__main__":
    app.run()
