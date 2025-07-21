# /// script
# requires-python = "<3.13,>=3.12"
# dependencies = [
#     "marimo",
#     "rdkit==2025.3.3",
# ]
# ///

import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from collections import defaultdict
    from itertools import cycle

    try:
        from rdkit.Chem import MolFromSmarts
        from rdkit.Chem import MolFromSmiles
        from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DSVG
        from rdkit.Chem.rdDepictor import Compute2DCoords

        message = mo.md("✅ Your environment supports **RDKit**, all good!")
    except ImportError:
        message = mo.md(
            "⚠️ **RDKit not available in this environment**.\n\n"
            "To run this script:\n"
            "```bash\n"
            "uvx marimo run https://raw.githubusercontent.com/Adafede/marimo/refs/heads/main/apps/mols.py\n"
            "```\n"
            "If using Docker, toggle **App View** (bottom right or `cmd + .`)."
        )
        Compute2DCoords = MolDraw2DSVG = MolFromSmarts = MolFromSmiles = None
    return (
        cycle,
        Compute2DCoords,
        defaultdict,
        message,
        mo,
        MolDraw2DSVG,
        MolFromSmarts,
        MolFromSmiles,
    )


@app.cell
def _(message):
    message
    return ()


@app.cell
def _(mo):
    smi_input = mo.ui.text_area(
        label="## Enter SMILES (one per line)",
        placeholder="e.g., CCO Ethanol",
        value="CC(=O)O acetic acid\nCCN(CC)CCN N,N-Diethylethylenediamine\nC[C@@H]1C=C(OC)C([C@@]2(C)[C@H]1C[C@@H]3[C@]4(C)[C@@H]2C(C(OC)=C(C)[C@@H]4CC(O3)=O)=O)=O quassin",
    )
    smi_input
    return (smi_input,)


@app.cell
def _(mo):
    smarts_input = mo.ui.text_area(
        label="## Enter SMARTS patterns (one per line)",
        placeholder="e.g., [OH] Hydroxyl",
        value="[#7] nitrogen\n[OH] hydroxyl\n[$(C);!$(C(~C)~C)]~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)(~C)~C)]1~[$(C);!$(C(~C)(~C)(~C)~C)](~[$(C);!$(C(~C)~C)])~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)(~C)~C)]2C3([$(C);!$(C(~C)~C)])~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)(~C)~C)](~[$(C);!$(C(~C)~C)])~[$(C);!$(C(~C)(~C)(~C)~C)]3~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)~C)]C2([$(C);!$(C(~C)~C)])1 picrasane",
    )
    smarts_input
    return (smarts_input,)


@app.cell
def _(mo, smarts_input):
    smarts_list = parse_input(smarts_input.value)
    toggles = {
        smarts: mo.ui.switch(value=True, label=name) for name, smarts in smarts_list
    }

    mo.md("## Toggle SMARTS Highlights")
    for switch in toggles.values():
        switch

    return (toggles,)


@app.cell
def _(mo):
    submit_button = mo.ui.button(label="🔬 Render Molecules")
    submit_button
    return (submit_button,)


# --- Utility Functions ---
@app.cell
def _():
    def parse_input(text):
        """
        Parses each line as (name, value) where value is first and name is after first space.
        If no space, name == value.
        """
        items = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if " " in line:
                val, name = line.split(" ", 1)
                items.append((name.strip(), val.strip()))
            else:
                # No name given
                items.append((line, line))
        return items

    def hex_to_rgb_float(hex_color):
        h = hex_color.lstrip("#")
        return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

    return parse_input, hex_to_rgb_float


# --- Main Rendering Logic ---
@app.cell
def _(
    Compute2DCoords,
    cycle,
    defaultdict,
    MolDraw2DSVG,
    MolFromSmarts,
    MolFromSmiles,
    smi_input,
    smarts_input,
    submit_button,
    toggles,
    parse_input,
    hex_to_rgb_float,
):
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

    smiles = parse_input(smi_input.value)  # (name, smiles)
    raw_smarts = parse_input(smarts_input.value)  # (name, smarts)

    # Filter by toggle
    active_smarts = [
        (name, smarts) for name, smarts in raw_smarts if toggles[smarts].value
    ]

    # Parse SMARTS with colors
    parsed_smarts = []
    for (name, smarts), color in zip(active_smarts, color_cycle):
        mol = MolFromSmarts(smarts)
        if mol:
            parsed_smarts.append((name, smarts, mol, hex_to_rgb_float(color)))

    # Rendering function
    def render_molecule(name, smi, smarts_mols, match_counter):
        mol = MolFromSmiles(smi)
        if not mol:
            return (
                f"<div style='color:red;'>🚫 Invalid SMILES: <code>{smi}</code></div>"
            )

        Compute2DCoords(mol)

        atom_ids = []
        colors = {}
        tooltips = []

        for s_name, smarts, smarts_mol, color in smarts_mols:
            matches = mol.GetSubstructMatches(smarts_mol)
            if matches:
                for match in matches:
                    atom_ids.extend(match)
                    for idx in match:
                        colors[idx] = color
                tooltips.append(f"✅ {s_name}: {len(matches)} match(es)")
                match_counter[s_name] += 1

        drawer = MolDraw2DSVG(300, 300)
        drawer.DrawMolecule(mol, highlightAtoms=atom_ids, highlightAtomColors=colors)
        drawer.FinishDrawing()

        label = (
            f"<strong>{name}</strong><br><code>{smi}</code>"
            if name != smi
            else f"<code>{smi}</code>"
        )

        return (
            "<div style='display:inline-block; margin:12px; text-align:center; "
            "border:1px solid #eee; padding:10px; border-radius:8px; "
            "box-shadow: 2px 2px 5px rgba(0,0,0,0.1);'>"
            f"{drawer.GetDrawingText()}<br>{label}<br>"
            f"<small>{'<br>'.join(tooltips)}</small></div>"
        )

    match_counter = defaultdict(int)

    if not smiles:
        html = "<p style='color:orange;'>⚠️ Please enter at least one SMILES string.</p>"
        summary_html = ""
    else:
        rendered = [
            render_molecule(name, smi, parsed_smarts, match_counter)
            for name, smi in smiles
        ]

        # Generate SMARTS match summary
        summary_items = [
            f"<div style='display:flex; justify-content:space-between; align-items:center; "
            f"padding:6px 12px; border-bottom:1px solid #eee;'>"
            f"<span style='font-size:0.9em; font-weight:500;'>{name}</span>"
            f"<span style='background:{highlight_palette[i % len(highlight_palette)]};"
            f" color:#fff; padding:2px 8px; border-radius:12px; font-size:0.8em;'>"
            f"{match_counter[name]} mol{'s' if match_counter[name] != 1 else ''}</span>"
            f"</div>"
            for i, (name, _) in enumerate(active_smarts)
        ]

        summary_html = (
            "<div style='margin: 16px auto; padding:12px 16px; max-width:600px; "
            "background: #fafafa; border: 1px solid #ddd; border-radius: 8px; "
            "box-shadow: 1px 1px 5px rgba(0,0,0,0.05); font-family: sans-serif;'>"
            "<div style='font-weight:bold; font-size:1.1em; margin-bottom:8px;'>🔎 SMARTS Match Summary</div>"
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
def _(html, mo):
    mo.Html(html)
    return


if __name__ == "__main__":
    app.run()
