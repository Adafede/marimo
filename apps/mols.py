import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    import itertools

    return Chem, itertools, mo, rdDepictor, rdMolDraw2D


@app.cell
def _(mo):
    smi_input = mo.ui.text_area(
        label="## Enter SMILES (one per row)",
        placeholder="e.g., CCO",
        value="CC(=O)O\nCCN(CC)CCN\nC[C@@H]1C=C(OC)C([C@@]2(C)[C@H]1C[C@@H]3[C@]4(C)[C@@H]2C(C(OC)=C(C)[C@@H]4CC(O3)=O)=O)=O",
    )
    smi_input
    return (smi_input,)


@app.cell
def _(mo):
    smarts_input = mo.ui.text_area(
        label="## Enter SMARTS (one per row)",
        placeholder="e.g., [OH]",
        value="[N]\n[OH]\n[$(C);!$(C(~C)~C)]~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)(~C)~C)]1~[$(C);!$(C(~C)(~C)(~C)~C)](~[$(C);!$(C(~C)~C)])~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)(~C)~C)]2C3([$(C);!$(C(~C)~C)])~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)(~C)~C)](~[$(C);!$(C(~C)~C)])~[$(C);!$(C(~C)(~C)(~C)~C)]3~[$(C);!$(C(~C)(~C)~C)]~[$(C);!$(C(~C)(~C)~C)]C2([$(C);!$(C(~C)~C)])1",
    )
    smarts_input
    return (smarts_input,)


@app.cell
def _(mo, smarts_input):
    smarts_list = [
        line.strip() for line in smarts_input.value.splitlines() if line.strip()
    ]
    toggles = {smarts: mo.ui.switch(value=True, label=smarts) for smarts in smarts_list}
    mo.md("## Toggle SMARTS Highlights")
    for toggle in toggles.values():
        toggle
    return (toggles,)


@app.cell
def _(mo):
    submit_button = mo.ui.button(label="🔬 Render Molecules with SMARTS Highlighting")
    submit_button
    return (submit_button,)


@app.cell
def _(
    Chem,
    itertools,
    rdDepictor,
    rdMolDraw2D,
    smarts_input,
    smi_input,
    submit_button,
    toggles,
):
    _ = submit_button.value

    # --- Paul Tol Light palette ---
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
    color_cycle = itertools.cycle(highlight_palette)

    def parse_input(text: str) -> list[str]:
        return [line.strip() for line in text.splitlines() if line.strip()]

    def hex_to_rgb_float(hexcolor: str) -> tuple[float, float, float]:
        h = hexcolor.lstrip("#")
        return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

    def parse_smarts(smarts_strs: list[str]) -> list[tuple[str, Chem.Mol, tuple]]:
        parsed = []
        for smarts in smarts_strs:
            mol = Chem.MolFromSmarts(smarts)
            if mol:
                color_hex = next(color_cycle)
                color_rgb = hex_to_rgb_float(color_hex)
                parsed.append((smarts, mol, color_rgb))
        return parsed

    def render_molecule(
        smi: str, smarts_mols: list[tuple[str, Chem.Mol, tuple]]
    ) -> str:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return (
                f"<div style='color:red;'>🚫 Invalid SMILES: <code>{smi}</code></div>"
            )

        rdDepictor.Compute2DCoords(mol)

        atom_ids = []
        colors = {}
        tooltips = []

        for smarts, smarts_mol, color in smarts_mols:
            matches = mol.GetSubstructMatches(smarts_mol)
            if matches:
                for match in matches:
                    atom_ids.extend(match)
                    for idx in match:
                        colors[idx] = color
                tooltips.append(f"✅ {smarts}: {len(matches)} match(es)")

        drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
        drawer.DrawMolecule(mol, highlightAtoms=atom_ids, highlightAtomColors=colors)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()

        return (
            f"<div style='display:inline-block; margin:12px; text-align:center; border:1px solid #eee; "
            f"padding:10px; border-radius:8px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);'>"
            f"{svg}<br><code>{smi}</code><br><small>{'<br>'.join(tooltips)}</small></div>"
        )

    smiles = parse_input(smi_input.value)
    raw_smarts = parse_input(smarts_input.value)

    # Use toggles to filter only active SMARTS
    active_smarts = [s for s in raw_smarts if toggles[s].value]
    parsed_smarts = parse_smarts(active_smarts)

    if not smiles:
        html = "<p style='color:orange;'>⚠️ Please enter at least one SMILES string.</p>"
    else:
        rendered = (render_molecule(smi, parsed_smarts) for smi in smiles)
        html = (
            "<div style='display:flex; flex-wrap:wrap; justify-content:center;'>"
            f"{''.join(rendered)}</div>"
        )
    return (html,)


@app.cell
def _(html, mo):
    mo.Html(html)
    return


if __name__ == "__main__":
    app.run()
