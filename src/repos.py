"""
This module provides utilities for managing Git repositories and listing Python files within them.
Functions:
"""

import os
import git
from .config import BASE_DIR


def clone_or_pull(repo_name: str, repo_url: str) -> str:
    """Clone the repo if not present, otherwise pull updates."""
    repo_path = os.path.join(BASE_DIR, repo_name)
    if os.path.isdir(repo_path):
        print(f"Updating {repo_name}")
        git.Repo(repo_path).remotes.origin.pull()
    else:
        print(f"Cloning {repo_name}")
        git.Repo.clone_from(repo_url, repo_path)
    return repo_path


def list_py_files(repo_path: str) -> list[str]:
    """Return all Python files in a repository."""
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(repo_path)
        for file in files if file.endswith(".py")
    ]
