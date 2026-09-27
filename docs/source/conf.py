# Sphinx configuration for Quantsmind-LLM documentation.
# Source: docs/source/*.rst -> HTML published to GitHub Pages (gh-pages branch).
# See docs/RULEBOOK.md section 4.

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "QuantsMind-LLM"
copyright = "2026, QuantsMind"
author = "QuantsMind"
release = "0.1.0"

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

