# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from datetime import datetime


project = "llm-chat-factory"
copyright = f"{datetime.now().year}, Antonio Pisani"
author = "Antonio Pisani"
release = "0.1.0b1"

os.environ["SPHINX_BUILD"] = "True"

# Add package to path for autodoc
import os
import sys
from pathlib import Path


# Get the project root directory (parent of docs directory)
docs_dir = Path(__file__).parent
project_root = docs_dir.parent
src_dir = project_root / "src"

# Add src to path if it exists
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

# Print for debugging (only during build)
print(f"Sphinx build: Adding {src_dir} to sys.path")
print(f"sys.path: {sys.path[:3]}")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",  # Creates summary tables for modules/classes
    "sphinx_sitemap",  # Generates sitemap for search engines
    "sphinx_tabs.tabs",  # For tabbed code examples
]

# Configure autosummary for API docs generation
autosummary_generate = True

# Configure autodoc
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

# Intersphinx configuration for external documentation
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

intersphinx_disabled_domains = []  # type: ignore
intersphinx_timeout = 30
intersphinx_cache_limit = 90  # days
intersphinx_disabled_reftypes = ["*"]

# Configure myst-parser
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

# Configure copybutton
copybutton_prompt_text = "$ "  # Remove terminal prompts
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"
copybutton_here_doc_delimiter = "EOT"

# Display todos by seeting to True
todo_include_todos = True

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#2980B9",
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#2980B9",
    "analytics_id": "",
    "canonical_url": "https://apisani1.github.io/chat-factory/",
}

# For sitemap
html_baseurl = "https://apisani1.github.io/chat-factory/"
sitemap_filename = "sitemap.xml"


# Reduce warning noise
nitpicky = False
suppress_warnings = [
    "myst.header",
    "ref.*",  # Suppress all reference warnings
]


# Add custom CSS
def setup(app):  # type: ignore
    app.add_css_file("custom.css")


from unittest.mock import MagicMock


# Autodoc core settings
autoclass_content = "both"  # Include both class and __init__ docstrings
autodoc_member_order = "bysource"  # Keep same order as in the source code

# Simple mocking for straightforward imports
autodoc_mock_imports = []  # type: ignore


# For more complex mocking where simple mocking isn't sufficient
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):  # type: ignore
        return MagicMock()


# Complex modules that need special handling
MOCK_MODULES = []  # type: ignore

# Only update sys.modules if there are modules to mock
if MOCK_MODULES:
    sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)
