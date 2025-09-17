#!/usr/bin/env python3
# 
# conf.py
# 
# Created by Nicolas Fricker on 08/31/2025.
# Copyright © 2025 Nicolas Fricker. All rights reserved.
# 

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

project = 'GradGraph'
copyright = '2025, Nicolas Fricker'
author = 'Nicolas Fricker'
release = '1.0.0'

root_doc = "index"

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": True,
    # 'exclude-members': 'from_config,get_config,build',
}

autodoc_typehints = 'description'
typehints_fully_qualified = True

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

autosummary_generate = True
autosummary_imported_members = True
autosummary_generate_overwrite = True

add_module_names = False

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 5,
    "titles_only": False,
}
# html_theme = 'alabaster'
html_static_path = ['_static']

