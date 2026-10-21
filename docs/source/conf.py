# Sphinx configuration for Taksh documentation.
# Source: docs/source/*.rst -> HTML published to GitHub Pages (gh-pages branch).
# See docs/RULEBOOK.md section 4.

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

from taksh import __version__

project = "Taksh"
copyright = "2026, Ajit Kumar"
author = "Ajit Kumar"
version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

master_doc = "index"
language = "en"
pygments_style = "sphinx"

html_theme = "sphinx_rtd_theme"

