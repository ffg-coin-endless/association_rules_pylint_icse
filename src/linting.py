"""
Runs pylint on all Python files in a specified repository and saves the report 
as both JSON and CSV files.
"""

import os
import subprocess
import json
import pandas as pd
from .repos import list_py_files


def run_pylint_on_repo(repo_name: str, repo_path: str) -> str | None:
    """Run pylint on all Python files of a repo and save report to CSV."""
    json_path = f"pylint_{repo_name.lower()}_report.json"
    csv_path = f"pylint_{repo_name.lower()}_report.csv"
    py_files = list_py_files(repo_path)

    if not py_files:
        print(f"No Python files found in {repo_name}.")
        return None

    print(f"Running pylint on {repo_name} ({len(py_files)} files)...")
    with open(json_path, "w", encoding="utf-8") as out_file:
        result = subprocess.run(
            ["pylint", "--output-format=json", "--exit-zero", *py_files],
            stdout=out_file,
            stderr=subprocess.PIPE,
            text=True,
            check=False  # explicitly allow non-zero exit codes
        )

    if result.stderr.strip():
        print(f"stderr for {repo_name}:\n{result.stderr[:300]}...")

    if not os.path.exists(json_path) or os.path.getsize(json_path) == 0:
        print(f"No JSON output for {repo_name}.")
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not data:
            raise ValueError("Empty or invalid JSON data.")
    except (json.JSONDecodeError, ValueError, OSError) as e:
        print(f"Failed to parse JSON for {repo_name}: {e}")
        return None

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path
