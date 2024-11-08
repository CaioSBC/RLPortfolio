# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# sys.path.insert(0, str(Path(__file__).resolve().parents[2]) + "/rlportfolio")
# sys.path.insert(0, str(Path(__file__).resolve().parents[2]) + "/rlportfolio/algorithm")
# sys.path.insert(0, str(Path(__file__).resolve().parents[2]) + "/rlportfolio/data")
# sys.path.insert(0, str(Path(__file__).resolve().parents[2]) + "/rlportfolio/environment")
# sys.path.insert(0, str(Path(__file__).resolve().parents[2]) + "/rlportfolio/policy")
# sys.path.insert(0, str(Path(__file__).resolve().parents[2]) + "/rlportfolio/utils")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "RLPortfolio"
copyright = "2024, Caio Costa"
author = "Caio Costa"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.coverage"
]

templates_path = ["_templates"]
exclude_patterns = []
add_module_names = False
autodoc_class_signature = "separated"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/rlportfolio_logo.png"
html_theme_options = {
    'logo_only': True,
    # 'display_version': True,
}