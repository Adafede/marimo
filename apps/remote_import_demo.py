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
    from pathlib import Path
    import importlib.util
    import re

    # === Import modules from /public ===
    def find_module_imports(file_path: Path, module_path: str) -> list[str]:
        """Find all imports from 'modules.*' in a Python file"""
        imports = []
        with open(file_path, "r") as f:
            content = f.read()

        # Find imports like: from modules.x.y import z
        pattern1 = r"from\s+(modules\.[\w.]+)\s+import"
        imports.extend(re.findall(pattern1, content))

        # Find imports like: import modules.x.y
        pattern2 = r"import\s+(modules\.[\w.]+)"
        imports.extend(re.findall(pattern2, content))

        # Find relative imports like: from .normalize import
        pattern3 = r"from\s+\.([\w.]+)\s+import"
        relative_imports = re.findall(pattern3, content)

        # Convert relative imports to absolute
        # e.g., if we're in modules.text.formula.parse and see .normalize
        # convert to modules.text.formula.normalize
        parent_module = ".".join(module_path.split(".")[:-1])
        for rel_import in relative_imports:
            absolute_import = f"{parent_module}.{rel_import}"
            imports.append(absolute_import)

        return list(set(imports))

    def create_package_hierarchy(module_path: str):
        """Create all parent package modules"""
        parts = module_path.split(".")
        for i in range(1, len(parts)):
            parent_name = ".".join(parts[:i])
            if parent_name not in sys.modules:
                parent_module = importlib.util.module_from_spec(
                    importlib.util.spec_from_loader(parent_name, loader=None)
                )
                # Set __path__ to make it a proper package
                parent_module.__path__ = []
                parent_module.__package__ = parent_name
                sys.modules[parent_name] = parent_module

    def load_module_from_public(
        module_path: str, base_path: Path = None, loaded: set = None
    ):
        """Recursively load a Python module and all its dependencies from public directory"""
        if loaded is None:
            loaded = set()

        # Skip if already loaded
        if module_path in loaded:
            return
        loaded.add(module_path)

        if base_path is None:
            base_path = mo.notebook_dir() / "public"

        if not base_path.exists():
            print(f"Warning: {base_path} does not exist")
            return

        # Convert module path to file path
        file_path = base_path / f"{module_path.replace('.', '/')}.py"
        if not file_path.exists():
            print(f"Warning: {file_path} does not exist")
            return

        # Find and load dependencies first
        dependencies = find_module_imports(file_path, module_path)
        if dependencies:
            print(f"  Found dependencies for {module_path}: {dependencies}")
        for dep in dependencies:
            load_module_from_public(dep, base_path, loaded)

        # Skip if already in sys.modules
        if module_path in sys.modules:
            return

        print(f"Loading module: {module_path}")

        # Create parent packages with __path__ attribute
        create_package_hierarchy(module_path)

        # Read the file content
        with open(file_path, "r") as f:
            module_code = f.read()

        # Create and execute the module
        spec = importlib.util.spec_from_loader(module_path, loader=None)
        module = importlib.util.module_from_spec(spec)
        module.__package__ = ".".join(module_path.split(".")[:-1])
        sys.modules[module_path] = module

        # Execute the module code
        try:
            exec(module_code, module.__dict__)
            print(f"  ✓ Successfully loaded {module_path}")
        except Exception as e:
            print(f"  ✗ Error loading {module_path}: {e}")
            import traceback

            traceback.print_exc()

    # Load modules (dependencies will be loaded automatically)
    load_module_from_public("modules.text.formula.parse")
    load_module_from_public("modules.chem.cdk.depict.url_from_smiles")

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
