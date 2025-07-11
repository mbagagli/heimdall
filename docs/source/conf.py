from __future__ import annotations

import importlib.metadata
import pathlib
import sys
from datetime import datetime

# ---------------------------------------------------------------------------#
# P A T H   S E T U P
# ---------------------------------------------------------------------------#
DOCS_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_DIR.parents[1]          # two levels up -> repo root
sys.path.insert(0, str(PROJECT_ROOT))       # so "import heimdall" works

# ---------------------------------------------------------------------------#
# P R O J E C T   I N F O
# ---------------------------------------------------------------------------#
project = "HEIMDALL"
author = "Matteo Bagagli"
copyright = f"{datetime.now().year}, {author}"

# Single-source the version
try:
    release = importlib.metadata.version("heimdall")      # installed
except importlib.metadata.PackageNotFoundError:
    # fall back to hard-coded dev version when building docs in a clean env
    release = "0.3.0"
version = release.split("+")[0]          # Short X.Y.Z without git hash

# ---------------------------------------------------------------------------#
# G E N E R A L   C O N F I G
# ---------------------------------------------------------------------------#
extensions = [
    # Core
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",

    # Nice-to-have
    "sphinx_autodoc_typehints",   # PEP 484 signatures
    "myst_parser",                # uncomment if you have Markdown files
]

autosummary_generate = True
autodoc_typehints = "description"            # show hints in the text, not sigs
autodoc_mock_imports = ["torch", "tensorflow"]   # add big opt deps here

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---------------------------------------------------------------------------#
# H T M L   O U T P U T
# ---------------------------------------------------------------------------#
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Optional: tweak RTD theme colours or add a logo
# html_theme_options = {"logo_only": True}
# html_logo = "_static/heimdall_logo.svg"
