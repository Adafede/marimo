# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "marimo",
#     "requests==2.32.5",
# ]
#
#
# [tool.marimo.display]
# theme = "system"
# ///

import marimo

__generated_with = "0.19.2"
app = marimo.App(app_title="Package Demo")

with app.setup:
    import sys
    import importlib.util
    from contextlib import contextmanager

    try:
        import requests
    except ImportError:
        raise ImportError("requests library is required. Install with: pip install requests")

    @contextmanager
    def remote_repo(url):
        """Context manager for importing from remote repo"""

        class RemoteImporter:
            def __init__(self, base_url):
                self.base_url = base_url.rstrip('/')

            def find_spec(self, fullname, path, target=None):
                parts = fullname.split('.')

                if len(parts) > 2:
                    return None

                return importlib.util.spec_from_loader(
                    fullname, 
                    self,
                    origin=f"{self.base_url}/{'/'.join(parts)}",
                    is_package=(len(parts) == 1)
                )

            def create_module(self, spec):
                return None

            def exec_module(self, module):
                fullname = module.__name__
                parts = fullname.split('.')

                if len(parts) == 1:
                    # Top-level package (text)
                    module_name = parts[0]
                    init_url = f"{self.base_url}/{module_name}/__init__.py"

                    try:
                        response = requests.get(init_url)
                        response.raise_for_status()
                        code = response.text

                        module.__path__ = [f"{self.base_url}/{module_name}"]
                        module.__package__ = module_name

                        exec(code, module.__dict__)
                    except Exception as e:
                        raise ImportError(f"Cannot load package {module_name}: {e}")

                elif len(parts) == 2:
                    # Submodule (text.smiles)
                    package_name, submodule_name = parts
                    submodule_url = f"{self.base_url}/{package_name}/{submodule_name}.py"

                    try:
                        response = requests.get(submodule_url)
                        response.raise_for_status()
                        code = response.text

                        module.__package__ = package_name
                        exec(code, module.__dict__)
                    except Exception as e:
                        raise ImportError(f"Cannot load submodule {fullname}: {e}")

        importer = RemoteImporter(url)
        sys.meta_path.insert(0, importer)
        try:
            yield
        finally:
            sys.meta_path.remove(importer)

    # Import the submodule, then import what you need from it
    with remote_repo(url="https://raw.githubusercontent.com/Adafede/marimo/main/modules"):
        from text.smiles import validate_smiles


@app.cell
def _():
    validate_smiles("aaa")
    return


if __name__ == "__main__":
    app.run()
