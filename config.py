"""
Configuration module for repository analysis and plotting.
This module defines:
- A dictionary of exemplar repositories (`REPOS`) to be analyzed, mapping 
    repository names to their GitHub URLs.
- The local base directory (`BASE_DIR`) where repositories will be cloned 
    or stored.
- Ensures the base directory exists.
- Sets global plotting styles for matplotlib visualizations, including 
    font sizes, figure size, and font families.
    Intended for use in scripts and notebooks that analyze Python 
    repositories and generate publication-quality plots.
"""

import os
import matplotlib as mpl

# Exemplar Repositories for analysis
REPOS = {
    "TheAlgorithms": "https://github.com/TheAlgorithms/Python.git",
    "Matplotlib": "https://github.com/matplotlib/matplotlib.git",
    "Sklearn": "https://github.com/scikit-learn/scikit-learn.git",
    "Flask": "https://github.com/pallets/flask.git",
    "Pymeasure": "https://github.com/pymeasure/pymeasure.git",
    "Requests": "https://github.com/psf/requests.git",
    "Numpy": "https://github.com/numpy/numpy.git",
}

# Local base directory for repositories
BASE_DIR = "repos"
os.makedirs(BASE_DIR, exist_ok=True)

# Global plotting style
mpl.rcParams.update({
    "font.size": 6,
    "axes.labelsize": 6,
    "axes.titlesize": 7,
    "legend.fontsize": 6,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "figure.figsize": (7, 3.5),
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "sans-serif"],
})
