# Minimal local Sphinx configuration for Quantsmind-LLM documentation.
# Not a GitHub Pages configuration; nothing is published externally.

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