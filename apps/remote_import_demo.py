# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "marimo",
#     "requests==2.32.5",
# ]
#
# [tool.marimo.display]
# theme = "system"
# ///

import marimo

__generated_with = "0.19.4"
app = marimo.App(app_title="Remote Import Demo")

with app.setup:
    import marimo as mo


@app.cell
def title():
    mo.md("""
    # Remote Import Demo
    """)
    return


@app.cell
def _():
    # === COPY INTO YOUR APP SETUP ===
    import sys

    # Toggle this flag for local vs remote development
    _USE_LOCAL = False  # Set to True for local development

    if _USE_LOCAL:
        # Add your local module directory to the path
        # Adjust this path to where your "modules" folder is located locally
        sys.path.insert(0, ".")

        def use(url):
            pass
    else:
        ## === GITHUB MODULES IMPORT ===
        import requests
        from importlib.machinery import ModuleSpec
        from importlib.abc import MetaPathFinder, Loader

        # Module content cache and existence cache
        _c = {}  # content cache: url -> source code
        _e = {}  # existence cache: url -> False (only caches non-existence)

        class _L(Loader):
            def __init__(s, b):
                s.b = b
                # Reusable session for connection pooling
                s.s = requests.Session()
                s.s.headers.update({"User-Agent": "marimo-remote-import"})

            def create_module(s, _):
                return None

            def exec_module(s, m):
                n, p = (
                    m.__spec__.name,
                    m.__spec__.submodule_search_locations is not None,
                )
                u = f"{s.b}/{n.replace('.', '/')}" + ("/__init__.py" if p else ".py")
                if u not in _c:
                    _c[u] = s.s.get(u).text
                m.__file__, m.__path__, m.__package__ = (
                    u,
                    [u.rsplit("/", 1)[0]] if p else None,
                    n if p else n.rpartition(".")[0],
                )
                exec(compile(_c[u], u, "exec"), m.__dict__)

        class _F(MetaPathFinder):
            def __init__(s, b, r):
                s.r, s.l = r, _L(b)

            def find_spec(s, n, *_):
                if n != s.r and not n.startswith(s.r + "."):
                    return None
                p = f"{s.l.b}/{n.replace('.', '/')}"
                py_url = p + ".py"
                init_url = p + "/__init__.py"

                # Check content cache first (already fetched)
                if py_url in _c:
                    return ModuleSpec(n, s.l, is_package=False)
                if init_url in _c:
                    return ModuleSpec(n, s.l, is_package=True)

                # Check non-existence cache
                if py_url in _e and init_url in _e:
                    return None

                # Try GET directly (skip HEAD - faster)
                if py_url not in _e:
                    try:
                        resp = s.l.s.get(py_url, timeout=10)
                        if resp.status_code == 200:
                            _c[py_url] = resp.text
                            return ModuleSpec(n, s.l, is_package=False)
                        _e[py_url] = False
                    except Exception:
                        _e[py_url] = False

                if init_url not in _e:
                    try:
                        resp = s.l.s.get(init_url, timeout=10)
                        if resp.status_code == 200:
                            _c[init_url] = resp.text
                            return ModuleSpec(n, s.l, is_package=True)
                        _e[init_url] = False
                    except Exception:
                        _e[init_url] = False

                return None

        def use(url):
            sys.meta_path.insert(0, _F(url.rsplit("/", 1)[0], url.rsplit("/", 1)[1]))

    # Usage: use("https://raw.githubusercontent.com/USER/REPO/BRANCH/PATH/PACKAGE")
    use("https://raw.githubusercontent.com/Adafede/marimo/main/modules")
    # === END ===

    mo.show_code()
    return


@app.cell
def imports():
    from modules.text.formula.parse import parse
    from modules.chem.cdk.depict.url_from_smiles import url_from_smiles

    mo.show_code()
    return parse, url_from_smiles


@app.cell
def example_1(parse):
    mo.show_code(parse("C₆H₁₂O₆"), position="above")
    return


@app.cell
def example_2(url_from_smiles):
    mo.show_code(mo.image(url_from_smiles("CCO")), position="above")
    return


if __name__ == "__main__":
    app.run()
