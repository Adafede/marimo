# /// script
# requires-python = "<3.13,>=3.12"
# dependencies = [
#     "marimo",
#     "rdkit==2025.3.3",
# ]
# ///

import marimo

__generated_with = "0.14.12"
app = marimo.App(app_title="Automated substructure depiction and verification")


@app.cell
def py_import():
    import marimo as mo
    from collections import defaultdict
    from itertools import cycle

    try:
        from rdkit.Chem import MolFromSmarts

        message = mo.md("✅ Your environment supports **RDKit**, all good!")
        rdkit_available = True
    except ImportError:
        message = mo.md(
            "⚠️ **RDKit not available in this environment**.\n\n"
            "To run this script:\n"
            "```bash\n"
            "uvx marimo run https://raw.githubusercontent.com/Adafede/marimo/refs/heads/main/apps/mols.py\n"
            "```\n"
            "If using Docker, toggle **App View** (bottom right or `cmd + .`)."
        )
        rdkit_available = False
        MolFromSmarts = None
    return (
        MolFromSmarts,
        cycle,
        defaultdict,
        message,
        mo,
        rdkit_available,
    )


@app.function
def parse_input(text):
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if " " in line:
            val, name = line.split(" ", 1)
            items.append((name.strip(), val.strip()))
        else:
            items.append((line, line))
    return items


@app.function
def hex_to_rgb_float(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


@app.function
def find_mcs_smarts(smiles_list):
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.rdFMCS import FindMCS

    mols = [MolFromSmiles(smi) for _, smi in smiles_list]
    mols = [mol for mol in mols if mol is not None]
    if len(mols) < 2:
        return None, "⚠️ Need at least two valid SMILES to find MCS."

    mcs_result = FindMCS(mols)
    if mcs_result.canceled or not mcs_result.smartsString:
        return None, "⚠️ Could not determine MCS."

    return mcs_result.smartsString, None


@app.function
def render_molecule(name, smi, smarts_mols, match_counter):
    from rdkit.Chem import MolFromSmiles
    from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DSVG
    from rdkit.Chem.rdDepictor import Compute2DCoords

    mol = MolFromSmiles(smi)
    if not mol:
        return f"<div style='color:red;'>🚫 Invalid SMILES: <code>{smi}</code></div>"

    Compute2DCoords(mol)
    atom_ids, colors, tooltips = [], {}, []

    for s_name, smarts, smarts_mol, color in smarts_mols:
        matches = mol.GetSubstructMatches(smarts_mol)
        if matches:
            for match in matches:
                atom_ids.extend(match)
                for idx in match:
                    colors[idx] = color
            tooltips.append(f"✅ {s_name}: {len(matches)} match(es)")
            match_counter[s_name] += 1

    drawer = MolDraw2DSVG(200, 200)
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


@app.cell
def message_md(message):
    message
    return


@app.cell
def stop_rdkit(mo, rdkit_available):
    if not rdkit_available:
        mo.stop()
    return


@app.cell
def input_smiles(mo, rdkit_available):
    if rdkit_available:
        smi_input = mo.ui.text_area(
            label="## Enter SMILES (one per line)",
            placeholder="e.g., CCO Ethanol",
            value="CC(=O)O acetic acid\nCCN(CC)CCN N,N-Diethylethylenediamine\nC[C@@H]1C=C(OC)C([C@@]2(C)[C@H]1C[C@@H]3[C@]4(C)[C@@H]2C(C(OC)=C(C)[C@@H]4CC(O3)=O)=O)=O quassin",
            full_width=True,
        )
    else:
        smi_input = None
    smi_input
    return (smi_input,)


@app.cell
def py_find_mcs(find_mcs_smarts, mo, parse_input, rdkit_available, smi_input):
    if rdkit_available:
        smiles_list = parse_input(smi_input.value)
        mcs_smarts, mcs_error = find_mcs_smarts(smiles_list)

        if mcs_smarts:
            mcs = mo.md(
                "### 📎 Automatically Detected Maximum Common Substructure (MCS) SMARTS\n\n"
                "The SMARTS pattern below was generated automatically. It may not always be chemically meaningful or appropriate for your use case, so please review it carefully.\n\n"
                "You can paste it below as a starting point:\n"
                f"```smarts\n{mcs_smarts}\n```"
            )
        elif mcs_error:
            mcs = mo.md(f"⚠️ {mcs_error}")
        else:
            mcs = mo.md("ℹ️ No MCS SMARTS generated.")
    else:
        mcs = None
    mcs
    return


@app.cell
def input_smarts(mo, rdkit_available):
    if rdkit_available:
        smarts_input = mo.ui.text_area(
            label="## Enter SMARTS patterns (one per line)",
            placeholder="e.g., [OH] Hydroxyl",
            value="[#7] nitrogen\n[OH] hydroxyl\n[$(C);!$(C(~C)~C)]~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)(~C)~C)]1~[$(C);!$(C(~C)(~C)(~C)~C)](~[$(C);!$(C(~C)~C)])~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)(~C)~C)]2C3([$(C);!$(C(~C)~C)])~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)(~C)~C)](~[$(C);!$(C(~C)~C)])~[$(C);!$(C(~C)(~C)(~C)~C)]3~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)~C)]C2([$(C);!$(C(~C)~C)])1 picrasane",
            full_width=True,
        )
    else:
        smarts_input = None
    smarts_input
    return (smarts_input,)


@app.cell
def input_toggle(mo, parse_input, smarts_input):
    smarts_list = parse_input(smarts_input.value)
    toggles = {
        smarts: mo.ui.switch(value=True, label=name) for name, smarts in smarts_list
    }

    mo.md("## Toggle SMARTS Highlights")
    for switch in toggles.values():
        switch
    return (toggles,)


@app.cell
def button_submit(mo, rdkit_available):
    if rdkit_available:
        submit_button = mo.ui.button(label="🔬 Render Molecules")
    else:
        submit_button = None
    submit_button
    return (submit_button,)


@app.cell
def py_generate_html(
    MolFromSmarts,
    cycle,
    defaultdict,
    hex_to_rgb_float,
    parse_input,
    smarts_input,
    smi_input,
    submit_button,
    toggles,
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

    smiles = parse_input(smi_input.value)
    raw_smarts = parse_input(smarts_input.value)

    active_smarts = [
        (name, smarts) for name, smarts in raw_smarts if toggles[smarts].value
    ]

    parsed_smarts = []
    for (name, smarts), color in zip(active_smarts, color_cycle):
        mol = MolFromSmarts(smarts)
        if mol:
            parsed_smarts.append((name, smarts, mol, hex_to_rgb_float(color)))

    match_counter = defaultdict(int)

    if not smiles:
        html = "<p style='color:orange;'>⚠️ Please enter at least one SMILES string.</p>"
    else:
        rendered = [
            render_molecule(name, smi, parsed_smarts, match_counter)
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
def html(html, mo):
    mo.Html(html)
    return


if __name__ == "__main__":
    app.run()
