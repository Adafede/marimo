# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "marimo",
#     "polars==1.40.1",
#     "rdkit==2025.9.5",
# ]
# ///

"""Marimo notebook demo for MCS analysis and CXSMILES inspection."""

import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    app_title="Automated substructure depiction and verification",
)

with app.setup:
    import sys

    import marimo as mo
    import polars as pl
    from rdkit import Chem

    sys.path.insert(0, ".")

    from modules.chem.rdkit.depict.collect_highlights import SmartsEntry
    from modules.chem.rdkit.depict.with_highlights import with_highlights
    from modules.chem.rdkit.smarts.parse import parse as parse_smarts
    from modules.chem.rdkit.smiles.parse import parse as parse_smiles


@app.cell
def _():
    mo.md("""
    # CX-YOGA: candidate-list to CX-SMILES (edge-aware)

    This notebook re-implements the workflow for practical cheminformatics use:

    - Accept either raw SMILES lists or a task JSON payload (`{cxsmiles, input_smiles}`)
    - Remove incoherent candidates using graph/fingerprint consistency filtering
    - Compute a strict MCS and map variable attachment points across candidates
    - Emit a first-pass CX-SMILES with `m:` blocks for positional variability
    - Canonicalize/hash/fingerprint molecules to compare candidate sets and CX inputs
    - Suggest repeating-unit candidates for potential `Sg:n` generalization

    Paste one of your PFAS / natural-product task payloads below.
    """)
    return


@app.cell
def _():
    smiles_input = mo.ui.text_area(
        label="Input task JSON or one SMILES per line",
        value='{"cxsmiles":"C1=CC=CC=C1C2=CC=CC=C2.Cl* |m:13:0.2.3|","input_smiles":["Clc1ccccc1-c2ccccc2","Clc1cccc(-c2ccccc2)c1","Clc1ccc(-c2ccccc2)cc1"]}',
        full_width=True,
    )
    smiles_input
    return (smiles_input,)


@app.cell
def _(smiles_input):
    import json

    from rdkit import DataStructs as _DataStructs
    from rdkit.Chem import rdMolDescriptors as _rdMolDescriptors

    _raw = smiles_input.value.strip()
    _provided_cxsmiles = None
    _benchmark_tasks = []
    _smiles_lines: list[str]
    if _raw.startswith("{") or _raw.startswith("["):
        try:
            _payload = json.loads(_raw)
            if isinstance(_payload, list):
                _benchmark_tasks = [t for t in _payload if isinstance(t, dict)]
            elif isinstance(_payload, dict) and isinstance(_payload.get("tasks"), list):
                _benchmark_tasks = [t for t in _payload["tasks"] if isinstance(t, dict)]
            elif isinstance(_payload, dict):
                _benchmark_tasks = [_payload]
        except json.JSONDecodeError:
            _benchmark_tasks = []

    if _benchmark_tasks:
        _first_task = _benchmark_tasks[0]
        _provided_cxsmiles = _first_task.get("cxsmiles")
        _smiles_lines = [
            str(_x).strip()
            for _x in _first_task.get("input_smiles", [])
            if str(_x).strip()
        ]
    else:
        _smiles_lines = [line.strip() for line in _raw.split("\n") if line.strip()]

    def _split_cx(_text: str) -> tuple[str, str | None]:
        _raw_text = _text.strip()
        if "|" not in _raw_text:
            return _raw_text, None
        _left, _right = _raw_text.split("|", 1)
        _meta = _right.rsplit("|", 1)[0].strip() if "|" in _right else _right.strip()
        return _left.strip(), (_meta if _meta else None)

    _records = []
    _failed = []
    for _smi in _smiles_lines:
        _base_smi, _cx_meta = _split_cx(_smi)
        _mol = parse_smiles(smiles=_base_smi)
        if _mol is None:
            _failed.append(_smi)
            continue
        _records.append(
            {
                "input_smiles": _smi,
                "input_base_smiles": _base_smi,
                "input_cx_meta": _cx_meta,
                "canonical_smiles": Chem.MolToSmiles(_mol, canonical=True),
                "mol": _mol,
            },
        )

    mo.stop(len(_records) == 0, mo.md("**[!] No valid SMILES provided.**"))

    _fps = [
        _rdMolDescriptors.GetMorganFingerprintAsBitVect(_r["mol"], 2, 2048)
        for _r in _records
    ]
    _threshold = 0.2
    _adj = {_idx: set() for _idx in range(len(_records))}
    for _idx in range(len(_records)):
        for _jdx in range(_idx + 1, len(_records)):
            _sim = _DataStructs.TanimotoSimilarity(_fps[_idx], _fps[_jdx])
            if _sim >= _threshold:
                _adj[_idx].add(_jdx)
                _adj[_jdx].add(_idx)

    _seen = set()
    _components = []
    for _node in _adj:
        if _node in _seen:
            continue
        _stack = [_node]
        _comp = []
        while _stack:
            _cur = _stack.pop()
            if _cur in _seen:
                continue
            _seen.add(_cur)
            _comp.append(_cur)
            _stack.extend(_adj[_cur] - _seen)
        _components.append(_comp)
    _largest_component = max(_components, key=len)
    _kept = set(_largest_component)

    for _i, _r in enumerate(_records):
        _r["coherent"] = _i in _kept

    mols = {
        "records": [_r for _i, _r in enumerate(_records) if _i in _kept],
        "provided_cxsmiles": _provided_cxsmiles,
        "failed_smiles": _failed,
        "coherence_threshold": _threshold,
        "benchmark_tasks": _benchmark_tasks,
    }

    _parsed_df = pl.DataFrame(
        {
            "input_smiles": [_r["input_smiles"] for _r in _records],
            "canonical_smiles": [_r["canonical_smiles"] for _r in _records],
            "coherent_cluster": [_r["coherent"] for _r in _records],
        },
    )
    mo.vstack(
        [
            mo.md(
                f"**Parsed {len(_records)} valid molecules; kept {len(mols['records'])} in the largest coherent component (Tanimoto >= {mols['coherence_threshold']:.2f}).**",
            ),
            _parsed_df,
            mo.md(f"**Benchmark tasks detected:** {len(_benchmark_tasks)}"),
            mo.md(f"**Failed parses:** {len(_failed)}"),
        ],
    )
    return (mols,)


@app.cell
def _(mols):
    from rdkit.Chem import rdFMCS

    _records = mols["records"]
    _mol_list = [r["mol"] for r in _records]
    mo.stop(len(_mol_list) == 0, mo.md("**[!] No coherent molecules to process.**"))

    if len(_mol_list) == 1:
        mcs_smarts = Chem.MolToSmarts(_mol_list[0])
        mcs_mol = Chem.MolFromSmarts(mcs_smarts)
        _mcs_num_atoms = _mol_list[0].GetNumAtoms()
    else:
        _mcs_result = rdFMCS.FindMCS(
            _mol_list,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=30,
        )
        mcs_smarts = _mcs_result.smartsString
        mcs_mol = parse_smarts(smarts=mcs_smarts) if mcs_smarts else None
        _mcs_num_atoms = _mcs_result.numAtoms

    mo.stop(
        mcs_mol is None,
        mo.md("**[!] No MCS found for the coherent candidate set.**"),
    )
    mo.vstack(
        [
            mo.md(f"**MCS atoms:** {_mcs_num_atoms}"),
            mo.md(f"**MCS SMARTS:** `{mcs_smarts}`"),
        ],
    )
    return mcs_mol, mcs_smarts


@app.cell
def _(mcs_mol, mcs_smarts, mols):
    _mcs_entry: SmartsEntry = (
        "MCS",
        mcs_smarts,
        mcs_mol,
        (0.2, 0.6, 1.0),
    )
    _grid = "".join(
        with_highlights(
            name=r["input_smiles"],
            smiles=r["input_smiles"],
            smarts_entries=[_mcs_entry],
            width=300,
            height=200,
        )
        for r in mols["records"]
    )
    mo.vstack(
        [
            mo.md("**MCS highlighted in blue:**"),
            mo.Html(f"<div style='display:flex; flex-wrap:wrap;'>{_grid}</div>"),
        ],
    )
    return


@app.cell
def _(mcs_mol, mcs_smarts, mols):
    def _best_match(input_mol):
        _matches = input_mol.GetSubstructMatches(mcs_mol, uniquify=True)
        if not _matches:
            return None
        _scored = []
        for _match in _matches:
            _core = set(_match)
            _edge_count = 0
            for _core_atom in _core:
                _edge_count += sum(
                    1
                    for _nb in input_mol.GetAtomWithIdx(_core_atom).GetNeighbors()
                    if _nb.GetIdx() not in _core
                )
            _scored.append((_edge_count, tuple(_match), _match))
        _scored.sort(key=lambda x: (-x[0], x[1]))
        return _scored[0][2]

    def _fragment_signature(mol, bond_idx, external_atom_idx):
        _frag = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=True)
        _frags = Chem.GetMolFrags(_frag)
        _target_atoms = None
        for _atom_ids in _frags:
            if external_atom_idx in _atom_ids:
                _target_atoms = list(_atom_ids)
                break
        if _target_atoms is None:
            return None
        _frag_smiles = Chem.MolFragmentToSmiles(
            _frag,
            atomsToUse=_target_atoms,
            canonical=True,
        )
        return _frag_smiles.replace("[*]", "*")

    _attachment_groups: dict[str, set[int]] = {}
    _representative_fragment: dict[str, str] = {}
    for _record in mols["records"]:
        _mol = _record["mol"]
        _match = _best_match(_mol)
        if _match is None:
            continue
        _mol_to_mcs = {atom_idx: i for i, atom_idx in enumerate(_match)}
        _core_atoms = set(_match)
        for _bond in _mol.GetBonds():
            _a = _bond.GetBeginAtomIdx()
            _b = _bond.GetEndAtomIdx()
            if (_a in _core_atoms) ^ (_b in _core_atoms):
                _core_atom = _a if _a in _core_atoms else _b
                _external_atom = _b if _a in _core_atoms else _a
                _sig = _fragment_signature(_mol, _bond.GetIdx(), _external_atom)
                if not _sig:
                    continue
                _attachment_groups.setdefault(_sig, set()).add(_mol_to_mcs[_core_atom])
                _representative_fragment.setdefault(_sig, _sig)

    _core_smiles = Chem.MolToSmiles(Chem.MolFromSmarts(mcs_smarts), canonical=False)
    _core_display_mol = Chem.MolFromSmiles(_core_smiles)
    _ranks = list(Chem.CanonicalRankAtoms(_core_display_mol, breakTies=False))

    def _expand_equivalent(_indices: set[int]) -> list[int]:
        _rank_set = {_ranks[i] for i in _indices if i < len(_ranks)}
        return sorted(i for i, r in enumerate(_ranks) if r in _rank_set)

    _components = [_core_smiles]
    _m_entries = []
    _offset = _core_display_mol.GetNumAtoms()
    for _sig in sorted(_attachment_groups):
        _fragment = _representative_fragment[_sig]
        _fragment_mol = Chem.MolFromSmiles(_fragment)
        if _fragment_mol is None:
            continue
        _components.append(_fragment)
        _star_atoms = [
            a.GetIdx() for a in _fragment_mol.GetAtoms() if a.GetAtomicNum() == 0
        ]
        _expanded = _expand_equivalent(_attachment_groups[_sig])
        for _star_local in _star_atoms:
            _m_entries.append(
                f"m:{_offset + _star_local}:{'.'.join(str(i) for i in _expanded)}",
            )
        _offset += _fragment_mol.GetNumAtoms()

    _generated_cx = ".".join(_components)
    if _m_entries:
        _generated_cx = f"{_generated_cx} |{','.join(_m_entries)}|"

    mols["generated_cxsmiles"] = _generated_cx

    _attachments_df = pl.DataFrame(
        {
            "fragment": list(sorted(_attachment_groups.keys())),
            "mcs_attachment_indices": [
                ".".join(str(i) for i in _expand_equivalent(_attachment_groups[k]))
                for k in sorted(_attachment_groups.keys())
            ],
        },
    )
    mo.vstack(
        [
            mo.md("**CX-YOGA v0 generated CX-SMILES:**"),
            mo.md(f"`{_generated_cx}`"),
            mo.md("**Attachment grouping used to assemble `m:` blocks:**"),
            _attachments_df,
        ],
    )
    return


@app.cell
def _(mols):
    import hashlib

    from rdkit import DataStructs as _DataStructs
    from rdkit.Chem import rdMolDescriptors as _rdMolDescriptors

    _records = mols["records"]
    _rows = []
    for _record in _records:
        _mol = _record["mol"]
        Chem.AssignStereochemistry(_mol, cleanIt=True, force=True)
        _canon = Chem.MolToSmiles(_mol, canonical=True)
        _cx = Chem.MolToCXSmiles(_mol)
        _hash = hashlib.sha256(_canon.encode("utf-8")).hexdigest()[:16]
        _rows.append(
            {
                "input_smiles": _record["input_smiles"],
                "canonical_smiles": _canon,
                "cxsmiles": _cx,
                "sha256_16": _hash,
            },
        )

    _fps = [
        _rdMolDescriptors.GetMorganFingerprintAsBitVect(_r["mol"], 2, 2048)
        for _r in _records
    ]
    _pairs = []
    for _idx in range(len(_fps)):
        for _jdx in range(_idx + 1, len(_fps)):
            _pairs.append(
                {
                    "pair": f"{_idx}-{_jdx}",
                    "tanimoto_ecfp4": round(
                        _DataStructs.TanimotoSimilarity(_fps[_idx], _fps[_jdx]),
                        4,
                    ),
                },
            )

    def _normalize_cx(_cx_text: str | None) -> str | None:
        if not _cx_text:
            return None
        _text = _cx_text.strip()
        _base = _text
        _meta = ""
        if "|" in _text:
            _left, _right = _text.split("|", 1)
            _base = _left.strip()
            _meta = (
                _right.rsplit("|", 1)[0].strip() if "|" in _right else _right.strip()
            )
        _base_mol = Chem.MolFromSmiles(_base)
        if _base_mol is None:
            return None
        _base_canon = Chem.MolToSmiles(_base_mol, canonical=True)
        if not _meta:
            return _base_canon

        _m_entries = []
        _sg_entries = []
        _other_entries = []
        for _token in [_t.strip() for _t in _meta.split(",") if _t.strip()]:
            if _token.startswith("m:"):
                _parts = _token.split(":")
                if len(_parts) == 3 and _parts[1].isdigit():
                    _anchor = int(_parts[1])
                    _idx = sorted(
                        {int(_v) for _v in _parts[2].split(".") if _v.isdigit()},
                    )
                    _m_entries.append((_anchor, tuple(_idx)))
                else:
                    _other_entries.append(_token)
            elif _token.startswith("Sg:"):
                _sg_entries.append(_token)
            else:
                _other_entries.append(_token)

        _m_entries = sorted(set(_m_entries))
        _sg_entries = sorted(set(_sg_entries))
        _other_entries = sorted(set(_other_entries))

        _m_norm = [f"m:{_a}:{'.'.join(str(_x) for _x in _i)}" for _a, _i in _m_entries]
        _all_meta = _m_norm + _sg_entries + _other_entries
        return f"{_base_canon} |{','.join(_all_meta)}|" if _all_meta else _base_canon

    def _generate_task_cx(_task_smiles: list[str]) -> tuple[str | None, str | None]:
        from rdkit.Chem import rdFMCS as _rdFMCS

        _mols_local = []
        for _s in _task_smiles:
            _m = parse_smiles(smiles=_s)
            if _m is not None:
                _mols_local.append(_m)
        if len(_mols_local) == 0:
            return None, "no_valid_smiles"

        if len(_mols_local) == 1:
            _mcs_smarts = Chem.MolToSmarts(_mols_local[0])
            _mcs_mol_local = Chem.MolFromSmarts(_mcs_smarts)
        else:
            _mcs_result_local = _rdFMCS.FindMCS(
                _mols_local,
                atomCompare=_rdFMCS.AtomCompare.CompareElements,
                bondCompare=_rdFMCS.BondCompare.CompareOrder,
                ringMatchesRingOnly=True,
                completeRingsOnly=True,
                timeout=30,
            )
            _mcs_smarts = _mcs_result_local.smartsString
            _mcs_mol_local = parse_smarts(smarts=_mcs_smarts) if _mcs_smarts else None
        if _mcs_mol_local is None:
            return None, "mcs_not_found"

        def _best_match_local(_input_mol):
            _matches_local = _input_mol.GetSubstructMatches(
                _mcs_mol_local,
                uniquify=True,
            )
            if not _matches_local:
                return None
            _scored_local = []
            for _match_local in _matches_local:
                _core_local = set(_match_local)
                _edge_count_local = 0
                for _core_atom_local in _core_local:
                    _edge_count_local += sum(
                        1
                        for _nb_local in _input_mol.GetAtomWithIdx(
                            _core_atom_local,
                        ).GetNeighbors()
                        if _nb_local.GetIdx() not in _core_local
                    )
                _scored_local.append(
                    (_edge_count_local, tuple(_match_local), _match_local),
                )
            _scored_local.sort(key=lambda _x: (-_x[0], _x[1]))
            return _scored_local[0][2]

        def _frag_sig_local(_mol_local, _bond_idx_local, _external_idx_local):
            _frag_local = Chem.FragmentOnBonds(
                _mol_local,
                [_bond_idx_local],
                addDummies=True,
            )
            _frags_local = Chem.GetMolFrags(_frag_local)
            _target_local = None
            for _atom_ids_local in _frags_local:
                if _external_idx_local in _atom_ids_local:
                    _target_local = list(_atom_ids_local)
                    break
            if _target_local is None:
                return None
            _frag_smiles_local = Chem.MolFragmentToSmiles(
                _frag_local,
                atomsToUse=_target_local,
                canonical=True,
            )
            return _frag_smiles_local.replace("[*]", "*")

        _attach_local: dict[str, set[int]] = {}
        for _mol_local in _mols_local:
            _match_local = _best_match_local(_mol_local)
            if _match_local is None:
                continue
            _map_local = {
                _atom_idx_local: _k for _k, _atom_idx_local in enumerate(_match_local)
            }
            _core_atoms_local = set(_match_local)
            for _bond_local in _mol_local.GetBonds():
                _a_local = _bond_local.GetBeginAtomIdx()
                _b_local = _bond_local.GetEndAtomIdx()
                if (_a_local in _core_atoms_local) ^ (_b_local in _core_atoms_local):
                    _core_atom_local = (
                        _a_local if _a_local in _core_atoms_local else _b_local
                    )
                    _external_local = (
                        _b_local if _a_local in _core_atoms_local else _a_local
                    )
                    _sig_local = _frag_sig_local(
                        _mol_local,
                        _bond_local.GetIdx(),
                        _external_local,
                    )
                    if _sig_local:
                        _attach_local.setdefault(_sig_local, set()).add(
                            _map_local[_core_atom_local],
                        )

        _core_smiles_local = Chem.MolToSmiles(
            Chem.MolFromSmarts(_mcs_smarts),
            canonical=False,
        )
        _core_display_local = Chem.MolFromSmiles(_core_smiles_local)
        if _core_display_local is None:
            return None, "mcs_core_not_renderable"
        _ranks_local = list(
            Chem.CanonicalRankAtoms(_core_display_local, breakTies=False),
        )

        def _expand_local(_indices_local: set[int]) -> list[int]:
            _rank_set_local = {
                _ranks_local[_idx_local]
                for _idx_local in _indices_local
                if _idx_local < len(_ranks_local)
            }
            return sorted(
                _idx_local
                for _idx_local, _rank_local in enumerate(_ranks_local)
                if _rank_local in _rank_set_local
            )

        _components_local = [_core_smiles_local]
        _m_entries_local = []
        _offset_local = _core_display_local.GetNumAtoms()
        for _sig_local in sorted(_attach_local):
            _frag_mol_local = Chem.MolFromSmiles(_sig_local)
            if _frag_mol_local is None:
                continue
            _components_local.append(_sig_local)
            _stars_local = [
                _a_local.GetIdx()
                for _a_local in _frag_mol_local.GetAtoms()
                if _a_local.GetAtomicNum() == 0
            ]
            _expanded_local = _expand_local(_attach_local[_sig_local])
            for _star_local in _stars_local:
                _m_entries_local.append(
                    f"m:{_offset_local + _star_local}:{'.'.join(str(_x) for _x in _expanded_local)}",
                )
            _offset_local += _frag_mol_local.GetNumAtoms()

        _cx_local = ".".join(_components_local)
        if _m_entries_local:
            _cx_local = f"{_cx_local} |{','.join(_m_entries_local)}|"
        return _cx_local, None

    _provided_cx = mols.get("provided_cxsmiles")
    _generated_cx = mols.get("generated_cxsmiles")
    _provided_norm = _normalize_cx(_provided_cx)
    _generated_norm = _normalize_cx(_generated_cx)
    _cx_compare_lines = []
    for _label, _cx in (("provided", _provided_cx), ("generated", _generated_cx)):
        if not _cx:
            continue
        _mol = Chem.MolFromSmiles(_cx.split("|")[0].strip())
        if _mol is None:
            _cx_compare_lines.append(
                f"- `{_label}` CX-SMILES could not be parsed by RDKit",
            )
            continue
        _canon_cx = Chem.MolToCXSmiles(_mol)
        _cx_hash = hashlib.sha256(_canon_cx.encode("utf-8")).hexdigest()
        _cx_compare_lines.append(f"- `{_label}` canonical CX hash: `{_cx_hash[:16]}`")
    if _provided_norm and _generated_norm:
        _cx_compare_lines.append(
            f"- normalized equivalence: `{_provided_norm == _generated_norm}`",
        )

    _benchmark_rows = []
    for _task_idx, _task in enumerate(mols.get("benchmark_tasks", [])):
        _task_name = str(_task.get("name") or f"task_{_task_idx + 1}")
        _task_category = str(_task.get("category") or "unspecified")
        _task_smiles = [
            str(_s).strip() for _s in _task.get("input_smiles", []) if str(_s).strip()
        ]
        _task_expected = _task.get("cxsmiles")
        _task_generated, _task_error = _generate_task_cx(_task_smiles)
        _task_expected_norm = _normalize_cx(_task_expected)
        _task_generated_norm = _normalize_cx(_task_generated)
        _task_pass = (
            _task_error is None
            and _task_expected_norm is not None
            and _task_generated_norm is not None
            and _task_expected_norm == _task_generated_norm
        )
        _reason = "ok" if _task_pass else (_task_error or "normalized_cx_mismatch")
        _benchmark_rows.append(
            {
                "task": _task_name,
                "category": _task_category,
                "n_input_smiles": len(_task_smiles),
                "pass": _task_pass,
                "reason": _reason,
                "expected_cxsmiles": _task_expected,
                "generated_cxsmiles": _task_generated,
                "expected_norm": _task_expected_norm,
                "generated_norm": _task_generated_norm,
            },
        )

    _cx_df = pl.DataFrame(_rows)
    _pair_df = (
        pl.DataFrame(_pairs)
        if _pairs
        else pl.DataFrame({"pair": [], "tanimoto_ecfp4": []})
    )
    _benchmark_df = (
        pl.DataFrame(_benchmark_rows)
        if _benchmark_rows
        else pl.DataFrame(
            {
                "task": [],
                "category": [],
                "n_input_smiles": [],
                "pass": [],
                "reason": [],
                "expected_cxsmiles": [],
                "generated_cxsmiles": [],
                "expected_norm": [],
                "generated_norm": [],
            },
        )
    )

    _benchmark_summary = "**Benchmark tasks:** none provided."
    if _benchmark_rows:
        _pass_count = sum(1 for _r in _benchmark_rows if _r["pass"])
        _total_count = len(_benchmark_rows)
        _benchmark_summary = f"**Benchmark tasks:** {_pass_count}/{_total_count} pass (strict normalized CX comparison)."

    mo.vstack(
        [
            mo.md("**Canonicalization, hashing, and per-molecule CX-SMILES:**"),
            _cx_df,
            mo.md("**Pairwise ECFP4 (Morgan r=2) similarities:**"),
            _pair_df,
            mo.md(
                "**CX-SMILES object-level comparison:**\n"
                + "\n".join(_cx_compare_lines)
                if _cx_compare_lines
                else "**CX-SMILES object-level comparison:** no provided/generated CX-SMILES available.",
            ),
            mo.md(_benchmark_summary),
            _benchmark_df,
        ],
    )
    return


@app.cell
def _(mols):
    _records = mols["records"]
    _cf2 = Chem.MolFromSmarts("C(F)(F)")

    _rows = []
    for _record in _records:
        _mol = _record["mol"]
        _c = sum(1 for _a in _mol.GetAtoms() if _a.GetAtomicNum() == 6)
        _f = sum(1 for _a in _mol.GetAtoms() if _a.GetAtomicNum() == 9)
        _o = sum(1 for _a in _mol.GetAtoms() if _a.GetAtomicNum() == 8)
        _cf2_count = len(_mol.GetSubstructMatches(_cf2))
        _rows.append(
            {
                "canonical_smiles": _record["canonical_smiles"],
                "C_atoms": _c,
                "F_atoms": _f,
                "O_atoms": _o,
                "CF2_motifs": _cf2_count,
            },
        )

    _pattern_df = pl.DataFrame(_rows).sort("C_atoms")
    _series = _pattern_df.get_column("C_atoms").to_list()
    _is_homologous = (
        len(_series) >= 3
        and len({_series[_idx + 1] - _series[_idx] for _idx in range(len(_series) - 1)})
        == 1
    )

    _suggestion = "No robust repeating-unit series detected for Sgroup generation."
    if _is_homologous:
        _step = _series[1] - _series[0]
        _mean_f_to_c = float(
            (
                _pattern_df.get_column("F_atoms") / _pattern_df.get_column("C_atoms")
            ).mean(),
        )
        _motif = "C(F)(F)" if _mean_f_to_c > 1.4 else "C"
        _base = _pattern_df.get_column("canonical_smiles")[0]
        _suggestion = f"Candidate repeating-unit annotation: `{_base} |Sg:n:3:n:ht|` (motif hint `{_motif}`, carbon step {int(_step)})."

    mo.vstack(
        [
            mo.md("**Repeating-unit diagnostics (for possible `Sg:n` annotation):**"),
            _pattern_df,
            mo.md(_suggestion),
        ],
    )
    return


@app.cell
def _():
    mo.md("""
    ## Reactive annotation engine (SMARTS + CXSMILES provenance)

    This section implements a reusable detection/decoration workflow:

    - CXSMILES-aware parsing with metadata preservation
    - precompiled SMARTS libraries with rule priorities
    - overlap-aware matching and deterministic conflict handling
    - traceable atom-level decoration with provenance tags
    - CXSMILES export with round-trip parse checks
    """)
    return


@app.cell
def _():
    rule_set_mode = mo.ui.dropdown(
        options=["default", "custom_json", "default_plus_custom"],
        value="default",
        label="Rule set mode",
    )
    allow_overlaps = mo.ui.checkbox(label="Allow overlapping matches", value=True)
    batch_size = mo.ui.number(value=128, label="Batch size", full_width=False)
    max_matches_per_rule = mo.ui.number(
        value=200,
        label="Max matches per rule and molecule",
        full_width=False,
    )
    use_parallel = mo.ui.checkbox(label="Parallel batches (experimental)", value=False)
    max_workers = mo.ui.number(value=4, label="Parallel workers", full_width=False)
    priority_direction = mo.ui.dropdown(
        options=["high_first", "low_first"],
        value="high_first",
        label="Priority direction",
    )
    custom_rules_json = mo.ui.text_area(
        label="Optional custom rules JSON list",
        value='[{"rule_id":"pfas_cf2","smarts":"C(F)(F)","category":"motif","priority":85,"label":"PFAS-CF2","color":"#d62728"}]',
        full_width=True,
    )

    mo.vstack(
        [
            mo.hstack(
                [
                    rule_set_mode,
                    priority_direction,
                    allow_overlaps,
                    use_parallel,
                ],
            ),
            mo.hstack([batch_size, max_matches_per_rule, max_workers]),
            custom_rules_json,
        ],
    )
    return (
        allow_overlaps,
        batch_size,
        custom_rules_json,
        max_matches_per_rule,
        max_workers,
        priority_direction,
        rule_set_mode,
        use_parallel,
    )


@app.cell
def _(custom_rules_json, rule_set_mode):
    import json as _json

    _default_rules = [
        {
            "rule_id": "fg_phenol_oh",
            "smarts": "[OX2H]-c",
            "category": "functional_group",
            "priority": 100,
            "label": "PhenolOH",
            "color": "#1f77b4",
        },
        {
            "rule_id": "fg_carboxylic_acid",
            "smarts": "C(=O)[OX2H1]",
            "category": "functional_group",
            "priority": 95,
            "label": "CO2H",
            "color": "#ff7f0e",
        },
        {
            "rule_id": "fg_sulfonic_acid",
            "smarts": "S(=O)(=O)[OX2H1,OX1-]",
            "category": "functional_group",
            "priority": 92,
            "label": "SO3H",
            "color": "#e377c2",
        },
        {
            "rule_id": "fg_nitro",
            "smarts": "[$([NX3](=O)=O),$([N+](=O)[O-])]",
            "category": "functional_group",
            "priority": 90,
            "label": "NO2",
            "color": "#17becf",
        },
        {
            "rule_id": "scaffold_biphenyl",
            "smarts": "c1ccccc1-c1ccccc1",
            "category": "scaffold",
            "priority": 80,
            "label": "BIPHENYL",
            "color": "#2ca02c",
        },
        {
            "rule_id": "motif_aryl_cl",
            "smarts": "[Cl]-c",
            "category": "reactive_motif",
            "priority": 70,
            "label": "ArCl",
            "color": "#9467bd",
        },
        {
            "rule_id": "motif_epoxide",
            "smarts": "[OX2r3][CX4r3][CX4r3]",
            "category": "reactive_motif",
            "priority": 65,
            "label": "EPOX",
            "color": "#bcbd22",
        },
    ]

    _custom_rules = []
    try:
        _parsed = (
            _json.loads(custom_rules_json.value.strip())
            if custom_rules_json.value.strip()
            else []
        )
        if isinstance(_parsed, list):
            _custom_rules = [r for r in _parsed if isinstance(r, dict)]
    except _json.JSONDecodeError:
        _custom_rules = []

    if rule_set_mode.value == "default":
        smarts_rules = _default_rules
    elif rule_set_mode.value == "custom_json":
        smarts_rules = _custom_rules
    else:
        smarts_rules = _default_rules + _custom_rules

    _clean_rules = []
    for _rule in smarts_rules:
        _rule_id = str(_rule.get("rule_id", "")).strip()
        _smarts = str(_rule.get("smarts", "")).strip()
        if not _rule_id or not _smarts:
            continue
        _clean_rules.append(
            {
                "rule_id": _rule_id,
                "smarts": _smarts,
                "category": str(_rule.get("category", "unspecified")),
                "priority": int(_rule.get("priority", 0)),
                "label": str(_rule.get("label", _rule_id)),
                "color": str(_rule.get("color", "#1f77b4")),
            },
        )

    _compiled = []
    _invalid = []
    for _rule in _clean_rules:
        _query = Chem.MolFromSmarts(_rule["smarts"])
        if _query is None:
            _invalid.append(_rule["rule_id"])
            continue
        _compiled.append({**_rule, "query": _query})

    _rules_df = (
        pl.DataFrame(_clean_rules)
        if _clean_rules
        else pl.DataFrame(
            {
                "rule_id": [],
                "smarts": [],
                "category": [],
                "priority": [],
                "label": [],
                "color": [],
            },
        )
    )
    compiled_smarts_rules = _compiled
    smarts_rules = _clean_rules

    mo.vstack(
        [
            mo.md(
                f"**Rules loaded:** {len(_clean_rules)} | compiled: {len(_compiled)} | invalid SMARTS: {len(_invalid)}",
            ),
            _rules_df,
            mo.md(
                f"**Invalid rule IDs:** {', '.join(_invalid) if _invalid else 'none'}",
            ),
        ],
    )
    return (compiled_smarts_rules,)


@app.cell
def _(
    allow_overlaps,
    batch_size,
    compiled_smarts_rules,
    max_matches_per_rule,
    max_workers,
    mols,
    priority_direction,
    use_parallel,
):
    from concurrent.futures import ThreadPoolExecutor
    import copy
    import hashlib as _hashlib
    import time

    _records = mols["records"]
    _rules = list(compiled_smarts_rules)
    _rules.sort(
        key=lambda _r: _r["priority"],
        reverse=(priority_direction.value == "high_first"),
    )

    def _merge_meta(_existing_meta: str | None, _new_meta_token: str) -> str:
        _tokens = []
        if _existing_meta:
            _tokens.extend(
                [_t.strip() for _t in _existing_meta.split(",") if _t.strip()],
            )
        _tokens.append(_new_meta_token)
        _tokens = sorted(set(_tokens))
        return ",".join(_tokens)

    def _annotate_record(_record: dict) -> dict:
        _mol = _record["mol"]
        _decorated = copy.deepcopy(_mol)
        _used_atoms = set()
        _atom_rules: dict[int, list[str]] = {}
        _matches = []

        for _rule in _rules:
            _all_matches = _mol.GetSubstructMatches(_rule["query"], uniquify=True)
            _count_for_rule = 0
            for _match in _all_matches:
                _match_atoms = tuple(int(_a) for _a in _match)
                if (not allow_overlaps.value) and any(
                    _a in _used_atoms for _a in _match_atoms
                ):
                    continue
                _count_for_rule += 1
                if _count_for_rule > int(max_matches_per_rule.value):
                    break
                _matches.append(
                    {
                        "rule_id": _rule["rule_id"],
                        "category": _rule["category"],
                        "priority": _rule["priority"],
                        "label": _rule["label"],
                        "atom_indices": _match_atoms,
                    },
                )
                _used_atoms.update(_match_atoms)
                for _atom_idx in _match_atoms:
                    _atom_rules.setdefault(_atom_idx, []).append(_rule["rule_id"])

        for _atom_idx, _rule_ids in _atom_rules.items():
            _atom = _decorated.GetAtomWithIdx(_atom_idx)
            _tag = "|".join(sorted(set(_rule_ids)))
            _atom.SetProp("cx_yoga.rule_ids", _tag)
            _atom.SetProp("atomLabel", _tag[:30])

        _decorated_cx = Chem.MolToCXSmiles(_decorated)
        _base_out = _decorated_cx.split("|")[0].strip()
        _base_out_mol = Chem.MolFromSmiles(_base_out)
        _round_trip_ok = _base_out_mol is not None

        _meta_token = f"cx_yoga:n_rules={len({m['rule_id'] for m in _matches})};n_matches={len(_matches)}"
        _merged_meta = _merge_meta(_record.get("input_cx_meta"), _meta_token)
        _decorated_with_meta = f"{_base_out} |{_merged_meta}|"

        return {
            "input_smiles": _record["input_smiles"],
            "canonical_smiles": _record["canonical_smiles"],
            "input_cx_meta": _record.get("input_cx_meta"),
            "decorated_cxsmiles": _decorated_cx,
            "decorated_cxsmiles_with_input_meta": _decorated_with_meta,
            "round_trip_ok": _round_trip_ok,
            "n_matches": len(_matches),
            "n_unique_rules": len({m["rule_id"] for m in _matches}),
            "match_rows": _matches,
            "decorated_mol": _decorated,
            "hash_sha256_16": _hashlib.sha256(
                _record["canonical_smiles"].encode("utf-8"),
            ).hexdigest()[:16],
        }

    _batch_n = max(1, int(batch_size.value))
    _batches = [
        _records[_start : _start + _batch_n]
        for _start in range(0, len(_records), _batch_n)
    ]
    _annotated = []
    _t0 = time.perf_counter()
    if use_parallel.value and len(_batches) > 1:
        _workers = max(1, int(max_workers.value))
        with ThreadPoolExecutor(max_workers=_workers) as _pool:
            for _batch_result in _pool.map(
                lambda _batch: [_annotate_record(_r) for _r in _batch],
                _batches,
            ):
                _annotated.extend(_batch_result)
    else:
        for _batch in _batches:
            _annotated.extend([_annotate_record(_r) for _r in _batch])
    _elapsed = time.perf_counter() - _t0

    _match_rows = []
    for _item in _annotated:
        for _m in _item["match_rows"]:
            _match_rows.append(
                {
                    "input_smiles": _item["input_smiles"],
                    "rule_id": _m["rule_id"],
                    "category": _m["category"],
                    "priority": _m["priority"],
                    "label": _m["label"],
                    "atom_indices": ".".join(str(_x) for _x in _m["atom_indices"]),
                },
            )

    _summary_df = (
        pl.DataFrame(
            {
                "input_smiles": [_x["input_smiles"] for _x in _annotated],
                "canonical_smiles": [_x["canonical_smiles"] for _x in _annotated],
                "n_matches": [_x["n_matches"] for _x in _annotated],
                "n_unique_rules": [_x["n_unique_rules"] for _x in _annotated],
                "round_trip_ok": [_x["round_trip_ok"] for _x in _annotated],
                "hash_sha256_16": [_x["hash_sha256_16"] for _x in _annotated],
            },
        )
        if _annotated
        else pl.DataFrame(
            {
                "input_smiles": [],
                "canonical_smiles": [],
                "n_matches": [],
                "n_unique_rules": [],
                "round_trip_ok": [],
                "hash_sha256_16": [],
            },
        )
    )
    _match_df = (
        pl.DataFrame(_match_rows)
        if _match_rows
        else pl.DataFrame(
            {
                "input_smiles": [],
                "rule_id": [],
                "category": [],
                "priority": [],
                "label": [],
                "atom_indices": [],
            },
        )
    )

    annotation_results = {
        "annotated": _annotated,
        "summary_df": _summary_df,
        "match_df": _match_df,
        "elapsed_s": _elapsed,
        "batch_count": len(_batches),
        "parallel": bool(use_parallel.value),
    }

    mo.vstack(
        [
            mo.md(
                f"**Annotation runtime:** {_elapsed:.4f}s | molecules: {len(_annotated)} | batches: {len(_batches)} | parallel: {bool(use_parallel.value)}",
            ),
            _summary_df,
            mo.md("**Match-level provenance table:**"),
            _match_df,
        ],
    )
    return (annotation_results,)


@app.cell
def _(annotation_results):
    _n = len(annotation_results["annotated"])
    _options = [str(_i) for _i in range(_n)]
    selected_molecule_index = mo.ui.dropdown(
        options=_options if _options else ["0"],
        value=_options[0] if _options else "0",
        label="Molecule index for before/after visualization",
    )
    selected_molecule_index
    return (selected_molecule_index,)


@app.cell
def _(annotation_results, selected_molecule_index):
    from rdkit.Chem import Draw as _Draw

    _annotated = annotation_results["annotated"]
    mo.stop(
        len(_annotated) == 0,
        mo.md("**No annotated molecules available for visualization.**"),
    )
    _idx = int(selected_molecule_index.value)
    _idx = max(0, min(_idx, len(_annotated) - 1))

    _row = _annotated[_idx]
    _before = Chem.MolFromSmiles(_row["canonical_smiles"])
    _after = _row["decorated_mol"]

    _highlight_atoms = set()
    for _m in _row["match_rows"]:
        _highlight_atoms.update(_m["atom_indices"])

    _img = _Draw.MolsToGridImage(
        [_before, _after],
        molsPerRow=2,
        subImgSize=(380, 280),
        legends=["Before", "Decorated"],
        highlightAtomLists=[list(_highlight_atoms), list(_highlight_atoms)],
        useSVG=False,
    )

    mo.vstack(
        [
            mo.md(f"**Selected molecule:** `{_row['input_smiles']}`"),
            mo.image(_img),
            mo.md(
                f"**Decorated CXSMILES:** `{_row['decorated_cxsmiles_with_input_meta']}`",
            ),
        ],
    )
    return


@app.cell
def _(compiled_smarts_rules):
    _rule_map = {r["rule_id"]: r for r in compiled_smarts_rules}
    _fixtures = [
        {
            "name": "salicylic_acid",
            "smiles": "O=C(O)c1ccccc1O",
            "expect_rule_ids": ["fg_phenol_oh", "fg_carboxylic_acid"],
        },
        {
            "name": "chlorobiphenyl",
            "smiles": "Clc1ccccc1-c2ccccc2",
            "expect_rule_ids": ["scaffold_biphenyl", "motif_aryl_cl"],
        },
        {
            "name": "nitrobenzene",
            "smiles": "O=[N+]([O-])c1ccccc1",
            "expect_rule_ids": ["fg_nitro"],
        },
    ]

    _rows = []
    for _fixture in _fixtures:
        _mol = Chem.MolFromSmiles(_fixture["smiles"])
        _detected = set()
        for _rid in _fixture["expect_rule_ids"]:
            _rule = _rule_map.get(_rid)
            if _rule is None:
                continue
            if _mol is not None and _mol.HasSubstructMatch(_rule["query"]):
                _detected.add(_rid)
        _expected = set(_fixture["expect_rule_ids"])
        _rows.append(
            {
                "fixture": _fixture["name"],
                "smiles": _fixture["smiles"],
                "expected_rules": ",".join(sorted(_expected)),
                "detected_rules": ",".join(sorted(_detected)),
                "pass": _expected.issubset(_detected),
            },
        )

    _fixtures_df = pl.DataFrame(_rows)
    _pass_n = int(_fixtures_df.get_column("pass").sum()) if _fixtures_df.height else 0
    _total_n = _fixtures_df.height
    mo.vstack(
        [
            mo.md(f"**Validation fixtures:** {_pass_n}/{_total_n} passed"),
            _fixtures_df,
        ],
    )
    return


if __name__ == "__main__":
    app.run()
