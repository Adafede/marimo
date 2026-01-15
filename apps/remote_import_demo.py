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

__generated_with = "0.19.2"
app = marimo.App(app_title="Remote Module Demo")

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
    # === GITHUB MODULES IMPORT: COPY INTO YOUR APP SETUP ===
    import sys, requests
    from importlib.machinery import ModuleSpec
    from importlib.abc import MetaPathFinder, Loader

    _c = {}


    class _L(Loader):
        def __init__(s, b):
            s.b, s.s = b, requests.Session()

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
            if (p + ".py") in _c or s.l.s.head(p + ".py").ok:
                return ModuleSpec(n, s.l, is_package=False)
            if (p + "/__init__.py") in _c or s.l.s.head(p + "/__init__.py").ok:
                return ModuleSpec(n, s.l, is_package=True)
            return None


    def use(url):
        sys.meta_path.insert(0, _F(url.rsplit("/", 1)[0], url.rsplit("/", 1)[1]))


    mo.show_code()
    return (use,)


@app.cell
def imports(use):
    # Usage: use("https://raw.githubusercontent.com/USER/REPO/BRANCH/PATH/PACKAGE")
    use("https://raw.githubusercontent.com/Adafede/marimo/main/modules")

    from modules.text.formula import parse_formula
    from modules.html.urls import structure_image_url

    mo.show_code()
    return parse_formula, structure_image_url


@app.cell
def example_1(parse_formula):
    mo.show_code(parse_formula("C6H12O6"), position="above")
    return


@app.cell
def example_2(structure_image_url):
    mo.show_code(mo.image(structure_image_url("CCO")), position="above")
    return


if __name__ == "__main__":
    app.run()
