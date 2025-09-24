"""
analysis.py
This module provides functions for mining frequent itemsets and association 
rules from pylint CSV reports, sorting and filtering rules by various metrics, 
and identifying shared or asymmetric rules across multiple repositories.
Functions
"""

import os
import ast
import pathlib
from pathlib import Path
import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


def analyze_pylint_report(csv_path: str, repo_name: str, out_dir: str = ".") -> None:
    """Mine frequent itemsets and association rules from a pylint CSV report."""

    os.makedirs(out_dir, exist_ok=True)

    print(f"\nReading {csv_path} for {repo_name}")
    df = pd.read_csv(csv_path)
    df.columns = [col.strip().replace(" ", "_") for col in df.columns]

    if "symbol" not in df.columns or "path" not in df.columns:
        print(f"Missing expected columns in {repo_name}")
        pd.DataFrame().to_csv(os.path.join(out_dir, f"rules_{repo_name}.csv"), index=False)
        return

    df = df.rename(columns={"message-id": "Error_Code", "path": "File"})
    error_counts = df["Error_Code"].value_counts()

    # Build transactions
    print(f"Building transactions for {repo_name}")
    transactions = df.groupby("File")["Error_Code"].apply(list).tolist()
    num_transactions = len(transactions)
    if num_transactions < 3:
        print(f"Too few transactions in {repo_name}. Skipping.")
        pd.DataFrame().to_csv(os.path.join(out_dir, f"rules_{repo_name}.csv"), index=False)
        return

    min_support = 3 / num_transactions
    print(f"Mining itemsets with min_support={min_support:.4f}")
    te = TransactionEncoder()
    trans_df = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_) # type: ignore

    frequent_itemsets = apriori(trans_df, min_support=min_support, use_colnames=True, max_len=3)
    if frequent_itemsets.empty:
        print(f"No frequent itemsets in {repo_name}")
        pd.DataFrame().to_csv(os.path.join(out_dir, f"rules_{repo_name}.csv"), index=False)
        return

    frequent_itemsets["itemsets"] = frequent_itemsets["itemsets"].apply(
        lambda x: frozenset(str(i) for i in x)
    )

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
    rules["Left_Hand_Side"] = rules["antecedents"].apply(
        lambda x: ", ".join(sorted(str(i) for i in x))
    )
    rules["Right_Hand_Side"] = rules["consequents"].apply(
        lambda x: ", ".join(sorted(str(i) for i in x)))
    rules["Jaccard"] = rules.apply(
        lambda row: row["support"] / (
            row["antecedent support"] + row["consequent support"] - row["support"]
        ),
        axis=1,
    )

    # Save rules
    rules_file = os.path.join(out_dir, f"rules_{repo_name}.csv")
    rules.to_csv(rules_file, index=False)

    # Save top rules graph
    top_rules = rules.sort_values(by="lift", ascending=False).head(20)
    if not top_rules.empty:
        graph = nx.DiGraph()
        for rule in top_rules.itertuples():
            for a in rule.antecedents: # type: ignore
                for c in rule.consequents: # type: ignore
                    graph.add_edge(str(a), str(c), lift=rule.lift, confidence=rule.confidence)

        node_sizes = [max(error_counts.get(n, 1), 2) * 40 for n in graph.nodes]
        edge_weights = [graph[u][v]["lift"] for u, v in graph.edges]
        if edge_weights:
            scaled_weights = [
                2 + 1.5 * (w - min(edge_weights)) / (max(edge_weights) - min(edge_weights) + 1e-5)
                for w in edge_weights
            ]
        else:
            scaled_weights = []

        pos = nx.shell_layout(graph)
        plt.figure(figsize=(3.5, 3.5))
        nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color="#A1C9F4",
                               alpha=0.9, edgecolors="k")
        nx.draw_networkx_edges(graph, pos, width=scaled_weights, edge_color="gray", # type: ignore
                               alpha=0.5, arrows=True)
        nx.draw_networkx_labels(graph, pos, font_size=7)
        plt.title(f"Top 20 Rules: {repo_name}", fontsize=9)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"assoc_network_{repo_name}.pdf"))
        plt.close()


def sort_rules_by_jaccard(repo_name: str, in_dir: Path, out_dir: Path) -> Path:
    """Load rules CSV, sort by Jaccard similarity (descending), and save."""
    in_file = Path(in_dir) / f"rules_{repo_name}.csv"
    out_file = Path(out_dir) / f"{repo_name.lower()}_sorted_by_jaccard.csv"

    df = pd.read_csv(in_file)

    # Add Jaccard if missing
    if (
        "Jaccard" not in df.columns
        and {"support", "antecedent support", "consequent support"} <= set(df.columns)
    ):
        df["Jaccard"] = df["support"] / (
            df["antecedent support"] + df["consequent support"] - df["support"]
        )

    df_sorted = df.sort_values(by="Jaccard", ascending=False)
    df_sorted.to_csv(out_file, index=False)

    print(f"Saved sorted rules to {out_file}")
    return out_file


def find_shared_one_to_one_rules(
    repos: list[str],
    in_dir: Path,
    min_conf: float = 0.5,
    min_lift: float = 2.0,
    min_support: float = 0.001,
) -> pd.DataFrame:
    """
    Find shared 1→1 rules across multiple repositories with thresholds applied.
    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        ['lhs', 'rhs', 'avg_confidence', 'avg_lift', 'avg_support'].
        Contains only rules that are shared across all repos.
    """
    filtered = {}
    sets = {}

    for name in repos:
        file_path = in_dir / f"rules_new_{name}.csv"
        df = pd.read_csv(file_path)

        # Restrict to 1→1 rules
        df = df[
            (df["antecedents"].apply(eval).apply(len) == 1) &
            (df["consequents"].apply(eval).apply(len) == 1)
        ]

        # Construct rule string
        df["rule_str"] = df["Left_Hand_Side"] + " → " + df["Right_Hand_Side"]
        df.set_index("rule_str", inplace=True)

        # Filter thresholds
        df = df[
            (df["confidence"] >= min_conf) &
            (df["lift"] >= min_lift) &
            (df["support"] >= min_support)
        ]

        filtered[name] = df
        sets[name] = set(df.index)

    # Shared rules across all repos
    shared_rules = set.intersection(*sets.values())
    if not shared_rules:
        return pd.DataFrame(columns=["lhs", "rhs", "avg_confidence", "avg_lift", "avg_support"])

    rows = []
    for rule in sorted(shared_rules):
        lhs, rhs = rule.split(" → ")

        confs = [filtered[repo].loc[rule, "confidence"] for repo in repos]
        lifts = [filtered[repo].loc[rule, "lift"] for repo in repos]
        supports = [filtered[repo].loc[rule, "support"] for repo in repos]

        rows.append({
            "lhs": lhs,
            "rhs": rhs,
            "avg_confidence": sum(confs) / len(repos),
            "avg_lift": sum(lifts) / len(repos),
            "avg_support": sum(supports) / len(repos),
        })

    return pd.DataFrame(rows)


def find_shared_one_to_one_rules_dynamic(
    repos: list[str],
    support_file: str,
    in_dir: str = ".",
    min_conf: float = 0.5,
    min_lift: float = 2.0
) -> pd.DataFrame:
    """
    Find shared 1→1 rules across repositories using dynamic min_support thresholds.

    Returns
    -------
    pd.DataFrame
        DataFrame of shared rules with average confidence, lift, and support.
    """
    in_dir = Path(in_dir) # type: ignore

    # Load dynamic min supports
    support_df = pd.read_csv(support_file)
    support_map = dict(zip(support_df['Repo'], support_df['MinSupport']))

    filtered = {}
    sets = {}

    for name in repos:
        csv_path = in_dir / f"rules_new_{name}.csv" # type: ignore
        df = pd.read_csv(csv_path)

        # Restrict to 1→1 rules
        df = df[
            (df['antecedents'].apply(eval).apply(len) == 1) &
            (df['consequents'].apply(eval).apply(len) == 1)
        ]

        df['rule_str'] = df['Left_Hand_Side'] + ' → ' + df['Right_Hand_Side']
        df.set_index('rule_str', inplace=True)

        min_support = support_map.get(name, 0.001)
        df = df[
            (df['confidence'] >= min_conf) &
            (df['lift'] >= min_lift) &
            (df['support'] >= min_support)
        ]

        filtered[name] = df
        sets[name] = set(df.index)

    # Shared rules across repos
    shared_rules = set.intersection(*sets.values())

    results = []

    def safe_get(rule, metric):
        values = []
        for repo in repos:
            if rule in filtered[repo].index:
                val = filtered[repo].loc[rule, metric]
                if isinstance(val, pd.Series):
                    values.append(val.iloc[0])
                else:
                    values.append(val)
        return values

    for rule in shared_rules:
        lhs, rhs = rule.split(' → ')

        confs = safe_get(rule, 'confidence')
        lifts = safe_get(rule, 'lift')
        supps = safe_get(rule, 'support')
        avg_conf = sum(confs) / len(confs)
        avg_lift = sum(lifts) / len(lifts)
        avg_support = sum(supps) / len(supps)

        results.append({
            "lhs": lhs,
            "rhs": rhs,
            "avg_confidence": round(avg_conf, 3),
            "avg_lift": round(avg_lift, 3),
            "avg_support": round(avg_support, 3),
        })

    return pd.DataFrame(results)


def parse_frozenset_column(col: pd.Series) -> pd.Series:
    """Robustly parse frozenset-like strings into Python frozensets."""
    def try_parse(x):
        if isinstance(x, frozenset):
            return x
        if isinstance(x, (list, set)):
            return frozenset(x)
        if not isinstance(x, str):
            return frozenset()
        if x.startswith("frozenset("):
            x = re.sub(r'^frozenset\((.*)\)$', r'\1', x)
        try:
            return frozenset(ast.literal_eval(x))
        except (ValueError, SyntaxError):
            return frozenset()
    return col.apply(try_parse)


def find_shared_error_rules_big3(repos, in_dir=".", max_antecedents=3, top_n=10):
    """
    Find shared rules with error consequents across the three big repos 
    (Matplotlib, Sklearn, Numpy).
    
    Returns
    -------
    pd.DataFrame
        Top shared rules with lift and support per repo.
    """
    in_dir = pathlib.Path(in_dir)

    error_pattern = re.compile(r"^E\d{4}$")
    filtered = {}
    sets = {}

    for name, filename in repos.items():
        df = pd.read_csv(in_dir / filename)
        df["antecedents"] = parse_frozenset_column(df["antecedents"])
        df["consequents"] = parse_frozenset_column(df["consequents"])

        df = df[
            (df["antecedents"].apply(len) <= max_antecedents) &
            (df["consequents"].apply(lambda x: any(error_pattern.match(e) for e in x)))
        ].copy()

        df["rule_key"] = df.apply(
            lambda row: (
            frozenset(row["antecedents"]),
            frozenset(row["consequents"])
            ),
            axis=1
        )
        filtered[name] = df
        sets[name] = set(df["rule_key"])

    # intersection across all repos
    shared = set.intersection(*sets.values())
    results = []

    for rule_key in shared:
        lhs, rhs = tuple(rule_key[0]), tuple(rule_key[1])
        rule_str = f"{lhs} → {rhs}"
        row = {"Rule": rule_str}

        max_lift = 0
        for repo in repos.keys():
            df = filtered[repo]
            match = df[df["rule_key"] == rule_key]
            if not match.empty:
                lift = match["lift"].values[0]
                supp = match["support"].values[0]
            else:
                lift, supp = 0.0, 0.0
            row[f"Lift_{repo}"] = lift # type: ignore
            row[f"Supp_{repo}"] = supp # type: ignore
            max_lift = max(max_lift, lift)

        row["MaxLift"] = max_lift # type: ignore
        results.append(row)

    df_out = (
        pd.DataFrame(results)
        .sort_values(by="MaxLift", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return df_out



def find_strong_asymmetries(path, min_asym=0.2):
    """
    Compute pairwise confidence asymmetries for 1→1 rules in a CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [A, B, confidence_AB, confidence_BA, asymmetry],
        sorted by |asymmetry| descending.
    """
    rules = pd.read_csv(path)

    # only 1→1 rules
    rules = rules[
        (rules["antecedents"].str.count(",") == 0) &
        (rules["consequents"].str.count(",") == 0)
    ].copy()

    rules["A"] = rules["Left_Hand_Side"].str.strip()
    rules["B"] = rules["Right_Hand_Side"].str.strip()
    rules["key"] = rules["A"] + "→" + rules["B"]
    rules["reverse_key"] = rules["B"] + "→" + rules["A"]

    # lookup reverse confidence
    reverse = rules[["key", "confidence"]].rename(
        columns={"key": "reverse_key", "confidence": "confidence_rev"}
    )
    merged = rules.merge(reverse, on="reverse_key", how="inner")

    merged["asymmetry"] = merged["confidence"] - merged["confidence_rev"]

    # filter and sort
    merged = merged.loc[merged["asymmetry"].abs() >= min_asym].copy()
    merged = merged[["A", "B", "confidence", "confidence_rev", "asymmetry"]]
    merged = merged.sort_values(by="asymmetry", key=np.abs, ascending=False).reset_index(drop=True)

    return merged
