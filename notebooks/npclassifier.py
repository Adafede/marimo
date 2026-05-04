# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "simple-parsing==0.1.8",
#     "polars==1.39.3",
#     "altair==6.0.0",
# ]
# ///

"""Marimo notebook for NPClassifier enrichment analysis and export."""

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="full")

with app.setup:
    import json
    import signal
    import sys
    import time
    import urllib.error
    import urllib.parse
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from dataclasses import dataclass, field
    from pathlib import Path

    import marimo as mo
    from simple_parsing import ArgumentParser

    CACHE_PATH = Path("apps/public/npclassifier/npclassifier_cache.json")

    @dataclass
    class Settings:
        """Command-line and runtime settings for NPClassifier requests."""

        smiles_file: str = field(
            default="",
            metadata={"help": "Path to input file with one SMILES per line"},
        )
        workers: int = field(
            default=8,
            metadata={"help": "Number of parallel HTTP workers"},
        )
        retries: int = field(
            default=3,
            metadata={"help": "Max retries per SMILES on transient errors"},
        )
        save_every: int = field(
            default=50,
            metadata={"help": "Flush cache to disk every N new results"},
        )

    _parser = ArgumentParser()
    _parser.add_arguments(Settings, dest="settings")

    def _parse_args() -> Settings:
        if mo.running_in_notebook():
            return Settings()
        return _parser.parse_args().settings

    settings = _parse_args()


@app.function
def load_cache(cache_path: Path = CACHE_PATH) -> dict:
    """Load cached NPClassifier responses from disk."""
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass
    return {}


@app.function
def save_cache(cache: dict, cache_path: Path = CACHE_PATH) -> None:
    """Atomic write via tmp-file rename. Skips 5xx errors so transient failures aren't persisted.

    Also write a CSV sidecar next to the JSON file.

    Parameters
    ----------
    cache : dict
        Cache.
    cache_path : Path
        CACHE_PATH. Default is CACHE_PATH.

    """
    clean = {k: v for k, v in cache.items() if not is_server_error(v)}

    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean, indent=2))
    tmp.replace(cache_path)

    csv_path = cache_path.with_suffix(".csv")
    tmp_csv = csv_path.with_suffix(".tmp")
    lines = ["smiles,pathway,superclass,class,isglycoside,error"]
    for smi, r in clean.items():

        def _join(key):
            return " $ ".join(r.get(key, []))

        def _esc(s):
            return f'"{s}"' if "," in s or '"' in s else s

        lines.append(
            ",".join(
                [
                    _esc(smi),
                    _esc(_join("pathway_results")),
                    _esc(_join("superclass_results")),
                    _esc(_join("class_results")),
                    _esc(str(r.get("isglycoside", ""))),
                    _esc(str(r.get("error", ""))),
                ],
            ),
        )
    tmp_csv.write_text("\n".join(lines))
    tmp_csv.replace(csv_path)


@app.function
def is_server_error(result: dict) -> bool:
    """Return whether a classification result reports an HTTP 5xx error."""
    err = result.get("error", "")
    return isinstance(err, str) and err.startswith("HTTP 5")


@app.function
def parse_smiles_file(text: str) -> list[str]:
    """Parse and sanitize SMILES candidates from raw text content."""
    import re

    _chem = re.compile(r"[CNOSPFBrIcnops(\[=@#]")
    _HEADERS = {
        "smiles",
        "smile",
        "structure",
        "inchi",
        "inchikey",
        "id",
        "name",
        "cid",
        "cas",
        "formula",
        "mw",
        "mol",
        "compound",
    }

    result = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) == 1:
            smi = parts[0].strip()
        else:
            smi = next(
                (
                    p.strip()
                    for p in parts
                    if _chem.search(p) and p.strip().lower() not in _HEADERS
                ),
                None,
            )
            if smi is None:
                continue
        if smi.lower() in _HEADERS or not _chem.search(smi):
            continue
        result.append(smi)
    return result


@app.function
def classify_one(smiles: str, retries: int = 3) -> tuple[str, dict]:
    """Classify one SMILES string using the NPClassifier API."""
    url = "https://npclassifier.gnps2.org/classify?smiles=" + urllib.parse.quote(
        smiles,
        safe="",
    )
    last_err = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return smiles, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                return smiles, {"error": f"HTTP {e.code}"}
            last_err = f"HTTP {e.code}"
        except urllib.error.URLError as e:
            last_err = f"URLError: {e.reason}"
        except TimeoutError:
            last_err = "timeout"
        except Exception as e:
            last_err = str(e)
        if attempt < retries - 1:
            time.sleep(2**attempt)
    return smiles, {"error": last_err}


@app.function
def classify_batch(
    smiles_list: list[str],
    cache: dict,
    cache_path: Path = CACHE_PATH,
    *,
    workers: int = 8,
    retries: int = 3,
    save_every: int = 50,
    progress_cb=None,
) -> dict:
    """Classify uncached SMILES strings in parallel and persist progress."""
    to_fetch = [s for s in smiles_list if s not in cache]
    if not to_fetch:
        return cache

    import threading

    _interrupted = False
    in_main = threading.current_thread() is threading.main_thread()

    def _flush_and_exit(signum, frame):
        nonlocal _interrupted
        _interrupted = True
        save_cache(cache, cache_path)
        print(
            f"\n[npclassifier] interrupted — {len(cache)} entries saved to {cache_path}",
            file=sys.stderr,
        )
        sys.exit(0)

    if in_main:
        prev_sigint = signal.signal(signal.SIGINT, _flush_and_exit)
        prev_sigterm = signal.signal(signal.SIGTERM, _flush_and_exit)

    try:
        pending = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(classify_one, s, retries): s for s in to_fetch}
            for future in as_completed(futures):
                if _interrupted:
                    break
                smi, result = future.result()
                cache[smi] = result
                pending += 1
                if progress_cb is not None:
                    progress_cb(increment=1)
                if pending >= save_every:
                    save_cache(cache, cache_path)
                    pending = 0
    finally:
        save_cache(cache, cache_path)
        if in_main:
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)

    return cache


@app.cell
def _show_header():
    mo.md(f"""
    # 🌿 NP Classifier — Batch SMILES Annotator

    Upload a file with one SMILES per line (or pass `--smiles_file` on the CLI).
    Results are cached at **`{CACHE_PATH}`** — any SMILES classified before won't be re-fetched.
    """)
    return


@app.cell
def _controls():
    file_input = mo.ui.file(label="SMILES file (.txt, one per line)", multiple=False)
    workers_slider = mo.ui.slider(1, 20, value=settings.workers, label="Workers")
    retries_slider = mo.ui.slider(1, 5, value=settings.retries, label="Retries")
    save_every_slider = mo.ui.slider(
        10,
        200,
        step=10,
        value=settings.save_every,
        label="Save every N",
    )
    mo.vstack(
        [
            file_input,
            mo.hstack([workers_slider, retries_slider, save_every_slider]),
        ],
    )
    return file_input, retries_slider, save_every_slider, workers_slider


@app.cell
def _load_smiles(file_input):
    import csv
    import re

    parse_details = {"mode": "none", "column": "", "delimiter": ""}
    smiles_list = []

    if file_input.value:
        _raw = file_input.value[0].contents.decode("utf-8", errors="replace")
        _lines = [line for line in _raw.splitlines() if line.strip()]
        _chem = re.compile(r"[CNOSPFBrIcnops()\[\]=@#\\/0-9+-]")
        _headers = {
            "smiles",
            "smile",
            "smiles_can",
            "smiles_canonical",
            "canonical_smiles",
            "smiles_iso",
            "smiles_isomeric",
            "isomeric_smiles",
            "structure",
            "compound_smiles",
        }

        def _smiles_like(value: str) -> bool:
            _v = value.strip()
            if not _v or " " in _v or _v.lower() in _headers:
                return False
            if _v.startswith("http"):
                return False
            return bool(_chem.search(_v))

        def _pick_delimiter(sample_lines: list[str]) -> str:
            _candidates = ["\t", ",", ";", "|"]
            _scores = {
                d: sum(line.count(d) for line in sample_lines[:50])
                / max(len(sample_lines[:50]), 1)
                for d in _candidates
            }
            _best = max(_scores, key=lambda k: _scores[k])
            return _best if _scores[_best] >= 0.5 else ""

        _delimiter = _pick_delimiter(_lines)
        parse_details = {"mode": "line", "column": "", "delimiter": _delimiter}

        if _delimiter:
            _rows = list(csv.reader(_lines, delimiter=_delimiter))
            _width = max((len(r) for r in _rows), default=0)
            _rows = [
                r + [""] * (_width - len(r))
                for r in _rows
                if any(cell.strip() for cell in r)
            ]

            if _rows and _width > 1:
                _header = [cell.strip() for cell in _rows[0]]
                _header_lc = [cell.lower() for cell in _header]
                _has_named_smiles = any(
                    h in _headers or "smiles" in h or h.endswith("_smi")
                    for h in _header_lc
                )
                _data_rows = _rows[1:] if _has_named_smiles else _rows

                _best_idx = None
                _best_score = float("-inf")
                for _idx in range(_width):
                    _name = _header_lc[_idx] if _idx < len(_header_lc) else ""
                    _values = [
                        row[_idx].strip()
                        for row in _data_rows[:300]
                        if _idx < len(row) and row[_idx].strip()
                    ]
                    if not _values:
                        continue
                    _like = sum(_smiles_like(v) for v in _values)
                    _ratio = _like / len(_values)
                    _score = _ratio * 10
                    if _name in _headers:
                        _score += 8
                    elif "smiles" in _name:
                        _score += 6
                    elif "smi" in _name or "structure" in _name:
                        _score += 3
                    if _score > _best_score:
                        _best_score = _score
                        _best_idx = _idx

                if _best_idx is not None and _best_score >= 4:
                    smiles_list = [
                        row[_best_idx].strip()
                        for row in _data_rows
                        if _best_idx < len(row) and _smiles_like(row[_best_idx])
                    ]
                    parse_details = {
                        "mode": "column",
                        "column": (
                            _header[_best_idx]
                            if _best_idx < len(_header)
                            else f"col_{_best_idx + 1}"
                        ),
                        "delimiter": _delimiter,
                    }

        if not smiles_list:
            smiles_list = parse_smiles_file(_raw)
    return parse_details, smiles_list


@app.cell
def _load_cache_cell():
    cache = load_cache(CACHE_PATH)
    return (cache,)


@app.cell
def _status(cache, parse_details, smiles_list):
    if not smiles_list:
        _status_md = mo.md("Upload a SMILES file to get started.")
    else:
        _n_new = len([s for s in smiles_list if s not in cache])
        _mode = parse_details.get("mode", "line")
        _column = parse_details.get("column", "")
        _delimiter = parse_details.get("delimiter", "")
        _source = "Detected SMILES column"
        if _mode != "column":
            _source = "Used line-based parser"
        _delimiter_label = "tab" if _delimiter == "\t" else (_delimiter or "none")
        _details = f"{_source}: `{_column or 'n/a'}` (delimiter: `{_delimiter_label}`)."
        _status_md = mo.md(
            f"**{len(smiles_list)} SMILES** loaded — "
            f"**{len(smiles_list) - _n_new}** cached, **{_n_new}** to fetch. "
            f"Cache: `{CACHE_PATH.resolve()}` ({len(cache)} entries).\n\n"
            f"{_details}",
        )
    _status_md
    return


@app.cell
def _run_button(smiles_list):
    mo.stop(not smiles_list)
    run_btn = mo.ui.run_button(label="Classify")
    run_btn
    return (run_btn,)


@app.cell
def _run_classify(
    cache,
    retries_slider,
    run_btn,
    save_every_slider,
    smiles_list,
    workers_slider,
):
    mo.stop(not smiles_list or not run_btn.value)
    _new = [s for s in smiles_list if s not in cache]

    with mo.status.progress_bar(total=len(_new) or 1, title="Classifying…") as _pb:
        results = classify_batch(
            smiles_list,
            cache,
            CACHE_PATH,
            workers=workers_slider.value,
            retries=retries_slider.value,
            save_every=save_every_slider.value,
            progress_cb=lambda increment: _pb.update(increment=increment),
        )
    mo.md(f"Done. Cache holds **{len(results)}** entries → `{CACHE_PATH.resolve()}`")
    return (results,)


@app.cell
def _results_table(results, run_btn, smiles_list):
    mo.stop(not smiles_list or not run_btn.value)
    import polars as pl

    _rows = []
    for _smi in smiles_list:
        _r = results.get(_smi, {})
        _rows.append(
            {
                "smiles": _smi,
                "pathway": ", ".join(_r.get("pathway_results", [])),
                "superclass": ", ".join(_r.get("superclass_results", [])),
                "class": ", ".join(_r.get("class_results", [])),
                "isglycoside": _r.get("isglycoside", ""),
                "error": _r.get("error", ""),
            },
        )

    results_df = pl.DataFrame(_rows)
    mo.ui.table(results_df, show_download=True)
    return pl, results_df


@app.cell
def _pathway_chart(pl, results_df, run_btn, smiles_list):
    mo.stop(not smiles_list or not run_btn.value)
    import altair as alt

    _counts = (
        results_df.filter(pl.col("pathway") != "")
        .with_columns(pl.col("pathway").str.split(", ").alias("pw"))
        .explode("pw")
        .rename({"pw": "Pathway"})
        .group_by("Pathway")
        .agg(pl.len().alias("Count"))
        .sort("Count", descending=True)
    )
    alt.Chart(_counts).mark_bar().encode(
        x=alt.X("Count:Q"),
        y=alt.Y("Pathway:N", sort="-x"),
        color=alt.Color("Pathway:N", legend=None),
        tooltip=["Pathway", "Count"],
    ).properties(title="Pathway Distribution", width=600, height=300)
    return


@app.cell
def _download(results_df, run_btn, smiles_list):
    mo.stop(not smiles_list or not run_btn.value)
    mo.download(
        data=results_df.write_csv().encode(),
        filename="npclassifier_results.csv",
        mimetype="text/csv",
        label="Download results CSV",
    )
    return


if __name__ == "__main__":
    if not settings.smiles_file:
        print("error: --smiles_file is required in CLI mode.", file=sys.stderr)
        sys.exit(1)

    _smiles = parse_smiles_file(Path(settings.smiles_file).read_text())
    _cache = load_cache(CACHE_PATH)
    _new = [s for s in _smiles if s not in _cache]
    print(
        f"[npclassifier] {len(_smiles)} SMILES | {len(_cache)} cached | {len(_new)} to fetch",
    )

    class _Counter:
        n = 0

        def tick(self, increment=1):
            self.n += increment
            print(f"\r[npclassifier] {self.n}/{len(_new)}", end="", flush=True)

    _counter = _Counter()
    _cache = classify_batch(
        _smiles,
        _cache,
        CACHE_PATH,
        workers=settings.workers,
        retries=settings.retries,
        save_every=settings.save_every,
        progress_cb=_counter.tick,
    )
    print(f"\n[npclassifier] done — {len(_cache)} entries in {CACHE_PATH.resolve()}")
