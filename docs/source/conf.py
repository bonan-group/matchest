"""Sphinx configuration for matchest documentation."""

import sys
from pathlib import Path

# The version is read from __about__.py
from matchest.__about__ import __version__

# Add source to path for autodoc
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

# -- Project information -----------------------------------------------------
project = "matchest"
copyright = "2025, Bonan Zhu"
author = "Bonan Zhu"


version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",  # Markdown support
    "sphinx.ext.autodoc",  # Auto-generate API docs
    "sphinx.ext.napoleon",  # Google/NumPy docstring support
    "sphinx.ext.intersphinx",  # Link to other projects
    "sphinx.ext.viewcode",  # Add links to source code
    "sphinx_click",  # CLI documentation
    "sphinx_copybutton",  # Copy button for code blocks
    "sphinx_design",  # Grid cards and UI components
]

# MyST parser settings
myst_enable_extensions = [
    "colon_fence",  # ::: fences
    "deflist",  # Definition lists
    "substitution",  # Variable substitution
]

# Templates and static files
templates_path = ["_templates"]
html_static_path = ["_static"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/bonan-group/matchest",
    "use_edit_page_button": False,
    "show_toc_level": 2,
    "navbar_align": "left",
    "navigation_depth": 3,
}

html_context = {"default_mode": "light"}

# -- Extension configuration -------------------------------------------------

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
    "pymatgen": ("https://pymatgen.org/", None),
    "aiida": ("https://aiida.readthedocs.io/projects/aiida-core/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Napoleon settings for docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_rtype = False
napoleon_use_param = True

# Copy button settings
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
