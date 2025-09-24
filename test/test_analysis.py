"""
Test suite for the `analysis` module in the `src` package.
This module contains unit tests for various functions that analyze pylint report 
data and association rules.
Tested Functions:
- analyze_pylint_report: Tests that the analysis function creates the expected 
    rules file from a minimal dataset.
- find_shared_one_to_one_rules: Tests detection of shared one-to-one association 
    rules across multiple repositories.
- find_shared_one_to_one_rules_dynamic: Tests detection of shared one-to-one 
    rules using dynamic support thresholds.
- find_shared_error_rules_big3: Tests identification of shared error rules among 
    three major repositories.
- find_strong_asymmetries: Tests detection of strong asymmetries in association 
    rules.
Each test uses temporary directories and files to ensure isolation and 
reproducibility.
"""

import pandas as pd
import pytest
from src import analysis


def test_analyze_pylint_report(tmp_path):
    """
    Test the analyze_pylint_report function by creating a minimal CSV report,
    running the analysis, and verifying that the expected output file is generated.
    """
    # Minimal dataset (3 transactions, overlapping error codes)
    data = [
        {"message-id": "E0001", "path": "file1.py", "symbol": "err"},
        {"message-id": "E0001", "path": "file2.py", "symbol": "err"},
        {"message-id": "E0002", "path": "file3.py", "symbol": "err"},
    ]
    csv_path = tmp_path / "report.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

    # Run analysis with output directed to tmp_path
    analysis.analyze_pylint_report(str(csv_path), "DummyRepo", out_dir=str(tmp_path))

    # Assert that rules file exists (may be empty)
    rules_file = tmp_path / "rules_DummyRepo.csv"
    assert rules_file.exists()


def test_find_shared_one_to_one_rules(tmp_path):
    """
    Test the `find_shared_one_to_one_rules` function to ensure it correctly 
    identifies and aggregates shared one-to-one association rules across 
    multiple repositories.
    """
    repos = ["RepoA", "RepoB", "RepoC"]

    def make_csv(path, lhs, rhs, support, confidence, lift):
        data = [{
            "antecedents": f"frozenset({{'{lhs}'}})",
            "consequents": f"frozenset({{'{rhs}'}})",
            "Left_Hand_Side": lhs,
            "Right_Hand_Side": rhs,
            "support": support,
            "confidence": confidence,
            "lift": lift,
            "antecedent support": support + 0.1,
            "consequent support": support + 0.2,
        }]
        pd.DataFrame(data).to_csv(path, index=False)

    for repo in repos:
        make_csv(tmp_path / f"rules_new_{repo}.csv", "E0001", "E0002", 0.3, 0.8, 2.5)

    # Run function
    df = analysis.find_shared_one_to_one_rules(repos, in_dir=tmp_path)

    # Assertions
    assert not df.empty
    assert (df["lhs"] == "E0001").any()
    assert (df["rhs"] == "E0002").any()
    assert df["avg_confidence"].iloc[0] == pytest.approx(0.8, rel=1e-9)
    assert df["avg_lift"].iloc[0] == 2.5




def test_find_shared_one_to_one_rules_dynamic(tmp_path):
    """
    Test the `find_shared_one_to_one_rules_dynamic` function to ensure it 
    correctly identifies shared one-to-one association rules across multiple 
    repositories using dynamically provided minimum support thresholds.
    """
    repos = ["RepoA", "RepoB"]
    support_file = tmp_path / "supports.csv"

    # Write dynamic support thresholds
    pd.DataFrame({"Repo": repos, "MinSupport": [0.2, 0.2]}).to_csv(support_file, index=False)

    def make_csv(path, lhs, rhs, support, confidence, lift):
        data = [{
            "antecedents": f"frozenset({{'{lhs}'}})",
            "consequents": f"frozenset({{'{rhs}'}})",
            "Left_Hand_Side": lhs,
            "Right_Hand_Side": rhs,
            "support": support,
            "confidence": confidence,
            "lift": lift,
        }]
        pd.DataFrame(data).to_csv(path, index=False)

    for repo in repos:
        make_csv(tmp_path / f"rules_new_{repo}.csv", "E0001", "E0002", 0.3, 0.8, 2.5)

    df = analysis.find_shared_one_to_one_rules_dynamic(repos, support_file, in_dir=tmp_path)

    assert not df.empty
    assert (df["lhs"] == "E0001").any()
    assert (df["rhs"] == "E0002").any()
    assert df["avg_confidence"].iloc[0] == pytest.approx(0.8, rel=1e-9)


def test_find_shared_error_rules_big3(tmp_path):
    """
    Test the `find_shared_error_rules_big3` function to ensure it correctly 
    identifies and returns shared error rules across three repositories 
    (Matplotlib, Sklearn, Numpy) using dummy CSV files. 
    """
    # Dummy repo files
    repos = {
        "Matplotlib": "rules_new_Matplotlib.csv",
        "Sklearn": "rules_new_Sklearn.csv",
        "Numpy": "rules_new_Numpy.csv"
    }

    # Helper to create dummy data
    def make_csv(path, lhs, rhs, lift, supp):
        df = pd.DataFrame([{
            "antecedents": f"frozenset({{'{lhs}'}})",
            "consequents": f"frozenset({{'{rhs}'}})",
            "Left_Hand_Side": lhs,
            "Right_Hand_Side": rhs,
            "support": supp,
            "confidence": 0.8,
            "lift": lift,
            "antecedent support": supp + 0.1,
            "consequent support": supp + 0.2,
        }])
        df.to_csv(path, index=False)

    for filename in repos.values():
        make_csv(tmp_path / filename, "A0001", "E0002", lift=2.5, supp=0.3)

    df_shared = analysis.find_shared_error_rules_big3(repos, in_dir=tmp_path, top_n=5)

    assert not df_shared.empty
    assert "Rule" in df_shared.columns
    assert "Lift_Sklearn" in df_shared.columns
    assert (df_shared["Rule"].str.contains("E0002")).any()


def test_find_strong_asymmetries(tmp_path):
    """
    Test the `find_strong_asymmetries` function to ensure it correctly 
    identifies strong asymmetric association rules.
    """
    csv_file = tmp_path / "rules.csv"

    # make dummy 1→1 rules
    df = pd.DataFrame([
        {"antecedents": "frozenset({'A'})", "consequents": "frozenset({'B'})",
         "Left_Hand_Side": "A", "Right_Hand_Side": "B", "confidence": 0.9},
        {"antecedents": "frozenset({'B'})", "consequents": "frozenset({'A'})",
         "Left_Hand_Side": "B", "Right_Hand_Side": "A", "confidence": 0.2},
    ])
    df.to_csv(csv_file, index=False)

    result = analysis.find_strong_asymmetries(csv_file, min_asym=0.5)

    assert not result.empty
    assert "asymmetry" in result.columns
    assert result.iloc[0]["A"] == "A"
    assert abs(result.iloc[0]["asymmetry"]) >= 0.5
