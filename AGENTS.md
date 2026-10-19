# Marimo WebAssembly Template — Agent Guide

This is a template repo for exporting marimo notebooks to static HTML/WebAssembly
and deploying to GitHub Pages / Codeberg. Apps use run mode; notebooks use edit mode.
Modules in `modules/` are inlined into apps at build time.

## Quick Start

```bash
# Install dependencies (uv is the package manager)
uv sync

# Export all notebooks and apps to _site/
uv run .github/scripts/build.py

# Or use the root symlink
uv run build.py

# Serve locally for testing
python -m http.server -d _site
```

> **Gotcha:** `uv run` may fail on macOS due to `loro` (a Rust dependency of
> `marimo`). If so, use globally-installed tools directly:
> - `ruff check .` and `ruff format .` (ruff is installed globally)
> - `graphify extract . --code-only --no-cluster --out .` (graphify is installed globally)

## Build / Test / Lint / Format

| Task | Command |
|------|---------|
| Export notebooks (build) | `uv run .github/scripts/build.py` |
| Lint | `ruff check .` |
| Format | `ruff format .` |
| Format check | `ruff format --check .` |
| Type check | `uv run ty check .` (may fail — see gotcha above) |
| Tests | No test suite exists yet |

Lint and format run via pre-commit hooks (`prek run`). The hooks also run
deptry (dependency check), typos (spell check), tombi (TOML format/lint),
pyupgrade, and zizmor (GitHub Actions security).

## Directory Structure

- `apps/` — Marimo apps (exported with `--mode run`, modules inlined)
- `notebooks/` — Marimo notebooks (exported with `--mode edit`, for display)
- `modules/` — Reusable Python modules (118 files across 13 subpackages)
- `templates/` — Jinja2 templates for the site index page
- `.github/scripts/build.py` — Build/export script (symlinked from root `build.py`)
- `prek.toml` — Pre-commit hooks configuration
- `pyproject.toml` — Project dependencies and tool config
- `uv.lock` — Pinned dependency versions

## Module Inlining

When exporting apps, the build script (`build.py`) inlines all `modules.*`
dependencies directly into each app's Python file before exporting to WASM.
This is done via regex-based import resolution and topological sorting.
Do not change module import patterns without verifying the build still works.

## Graphify Knowledge Graph

A graphify knowledge graph is available for code navigation:

```bash
# Regenerate the graph (requires graphify v0.9.90+)
graphify extract . --code-only --no-cluster --out .

# Query the graph
graphify explain "build.py"
graphify path "AGENTS.md" "build.py"
```

Output is written to `graphify-out/` (gitignored). The graph covers all tracked
Python files, configs, and templates.

## Conventions

- Notebook scripts use PEP 723 inline dependencies (`# /// script` block)
- Module functions follow numpydoc-style docstrings
- `.kb/` directory holds informal knowledge base notes (gitignored)
- `.agentignore` controls what graphify excludes from the graph
- `AGENTS.md` is tracked and is the canonical instruction file

## Do NOT Touch

- `graphify-out/` — auto-generated, regenerable
- `_site/` — build output
- `uv.lock` — managed by `uv`
- `AGENTS.md` content in this repo is the canonical instruction file (do not
  replace with notebook-creation instructions)

## Verification

Before completing a change, run:
1. `ruff check .` — must pass (76 pre-existing errors; see [tool.ruff] in pyproject.toml)
   - Some errors in `notebooks/`, `apps/`, and `modules/` are intentionally ignored via
     per-file rules due to marimo's reactive programming patterns
2. `uv run .github/scripts/build.py` — build must succeed (may fail on loro/Rust)
   - Alternative: `ruff check .` and `ruff format --check .` verify basic correctness

## Agent Configuration

Contributors using Claude Code may create `.claude/settings.json` locally with
permissions to allow `ruff check`, `ruff format`, `git` operations, and `graphify`
commands while denying destructive actions like `rm -rf`, `git push`, and
`git remote` modifications.
