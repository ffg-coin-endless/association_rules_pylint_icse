"""
This module defines a pipeline for analyzing Python repositories using pylint.
"""

import multiprocessing
import concurrent.futures
from .config import REPOS
from .repos import clone_or_pull
from .linting import run_pylint_on_repo
from .analysis import analyze_pylint_report
from .visualization import plot_grouped_rule_matrix


def run_pipeline(repo_name: str, repo_url: str) -> None:
    """Full pipeline for a single repository."""
    repo_path = clone_or_pull(repo_name, repo_url)
    csv_path = run_pylint_on_repo(repo_name, repo_path)
    if csv_path:
        analyze_pylint_report(csv_path, repo_name)
        plot_grouped_rule_matrix(repo_name)


def main() -> None:
    """Entry point for running the full pipeline across all repositories."""
    print(f"Using {multiprocessing.cpu_count()} cores")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_pipeline, name, url) for name, url in REPOS.items()]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except (
                concurrent.futures.process.BrokenProcessPool,
                concurrent.futures.TimeoutError
            ) as e:
                print(f"Process pool or timeout error during analysis: {e}")
            except OSError as e:
                # Handle OS-related errors (e.g., file not found, permission denied)
                print(f"OS error during analysis: {e}")
            except ValueError as e:
                # Handle value errors (e.g., bad input data)
                print(f"Value error during analysis: {e}")


if __name__ == "__main__":
    main()
