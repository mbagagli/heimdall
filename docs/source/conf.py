# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath("../.."))

project = 'HEIMDALL'
copyright = '2024, Matteo Bagagli'
author = 'Matteo Bagagli'
release = 'v0.3.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

extensions = [
    "sphinx.ext.autodoc",      # pull in docstrings
    "sphinx.ext.napoleon",     # Google- / NumPy-style parsing
    "sphinx.ext.autosummary",  # create one page per object
    "sphinx.ext.viewcode",     # add links to highlighted source
]

autosummary_generate = True        # build *.rst stubs automatically
napoleon_google_docstring = True   # you already set this
napoleon_use_param = True
