# marimo WebAssembly + GitHub Pages Template

This template repository demonstrates how to export [marimo](https://marimo.io)
notebooks to WebAssembly and deploy them to GitHub Pages.

## 📚 Included Examples

- `apps/bayesian_chemotaxonomic_profiler.py`: Bayesian chemotaxonomic profiling
- `apps/lotus_exporter.py`: Export compound data from LOTUS
- `apps/lotus_pubchem_tree.py`: Build taxonomic trees from PubChem
- `apps/lotus_wikidata_explorer.py`: Explore Wikidata for compound data
- `apps/mols.py`: Molecular structure utilities
- `notebooks/fibonacci.py`: Interactive Fibonacci sequence calculator
- `notebooks/npclassifier.py`: NPClassifier demo notebook

## 🚀 Usage

1. Fork this repository
2. Add your marimo files to the `notebooks/` or `apps/` directory
   1. `notebooks/` notebooks are exported with `--mode edit`
   2. `apps/` notebooks are exported with `--mode run`
3. Push to main branch
4. Go to repository **Settings > Pages** and change the "Source" dropdown to
   "GitHub Actions"
5. GitHub Actions will automatically build and deploy to Pages

## Including data or assets

To include data or assets in your notebooks, add them to the `public/`
directory.

For example, a notebook can load an image from `public/` via:

```python
import marimo as mo
img = mo.ui.file(label="Upload image")

# Load asset using notebook_location
df = pl.read_csv(mo.notebook_location() / "public" / "your_data.csv")
```

## 🎨 Templates

This repository includes two example templates for the generated site:

1. `index.html.j2` (default): A template with styling and a footer
2. `tailwind.html.j2`: A minimal and lean template using Tailwind CSS

To use a specific template, pass the `--template` parameter to the build script:

```bash
uv run .github/scripts/build.py --template templates/tailwind.html.j2
```

You can also create your own custom templates. See the
[templates/README.md](templates/README.md) for more information.

## 🧪 Testing

To test the export process, run `.github/scripts/build.py` from the root
directory.

```bash
uv run .github/scripts/build.py
```

This will export all notebooks in a folder called `_site/` in the root
directory. Then to serve the site, run:

```bash
python -m http.server -d _site
```

This will serve the site at `http://localhost:8000`.
