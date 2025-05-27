import os
import sys
from datetime import datetime
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# Add project root and src directory to sys.path for autodoc
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# -- Project information -----------------------------------------------------
project = "news-classification"
author = "news-classification contributors"
copyright = f"{datetime.now().year}, {author}"
release = "1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinxcontrib.mermaid",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_title = f"{project} {release} Documentation"
html_static_dir = Path(__file__).parent / "_static"
html_static_path = ["_static"] if html_static_dir.exists() else []

# -- Options for autodoc -----------------------------------------------------
add_module_names = False 