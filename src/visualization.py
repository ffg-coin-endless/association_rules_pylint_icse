"""
Visualization utilities for association rule mining results.
This module provides a collection of functions to visualize association rules 
mined from code repositories, including grouped rule matrices, scatter plots, 
bubble charts, UpSet plots, network graphs, and asymmetry matrices.
The visualizations are tailored for analyzing relationships between code 
warning/error messages and their severities.
"""

import os
import ast
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

import seaborn as sns
import networkx as nx
from upsetplot import from_contents, plot as upset_plot

matplotlib.use("Agg")

def plot_grouped_rule_matrix(repo_name: str, in_dir=".", out_dir="."):
    """Generates and saves a grouped association rule matrix plot for a given repository."""
    csv_file = os.path.join(in_dir, f"rules_{repo_name}.csv")
    if not os.path.exists(csv_file):
        print(f"Missing rules file: {csv_file}")
        return None

    df = pd.read_csv(csv_file)

    def safe_frozenset_parse(x):
        try:
            return frozenset(ast.literal_eval(x))
        except (ValueError, SyntaxError):
            return frozenset()

    df["antecedents"] = df["antecedents"].apply(safe_frozenset_parse)
    df["consequents"] = df["consequents"].apply(safe_frozenset_parse)

    df["lhs_label"] = df["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    df["rhs_label"] = df["consequents"].apply(lambda x: ", ".join(sorted(x)))

    grouped = df.groupby(["lhs_label", "rhs_label"]).agg({
        "lift": "max",
        "support": "sum",
    }).reset_index()

    pivot = grouped.pivot(index="lhs_label", columns="rhs_label", values="lift").fillna(0)

    size_map = defaultdict(float)
    for _, row in grouped.iterrows():
        key = (row["lhs_label"], row["rhs_label"])
        size_map[key] += row["support"] * 20

    _, ax = plt.subplots(figsize=(7, 5))
    for (i, lhs) in enumerate(pivot.index):
        for (j, rhs) in enumerate(pivot.columns):
            lift = pivot.loc[lhs, rhs]
            if lift == 0:
                continue
            size = size_map[(lhs, rhs)]
            ax.scatter(j, i, s=size, c=[[lift]], cmap="viridis", edgecolors="k") # type: ignore

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=90)
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Grouped Association Rules: {repo_name}")
    ax.set_xlabel("RHS Error Code")
    ax.set_ylabel("LHS Group")
    sm = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(  # type: ignore
            vmin=pivot.values.min(),
            vmax=pivot.values.max()
        )
    )
    plt.colorbar(sm, ax=ax, label="Lift")
    plt.tight_layout()

    out_file = os.path.join(out_dir, f"grouped_rule_matrix_{repo_name.lower()}.pdf")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")
    return out_file



def plot_lift_vs_jaccard(repo_name, in_dir=".", out_dir="."):
    """
    Scatter plot: Lift vs Jaccard, bubble size = confidence, color = support.
    """
    in_file = os.path.join(in_dir, f"rules_{repo_name}.csv")
    if not os.path.exists(in_file):
        print(f"Missing rules file: {in_file}")
        return None

    df = pd.read_csv(in_file)

    if "jaccard" not in df.columns:
        if {"support", "antecedent support", "consequent support"}.issubset(df.columns):
            df["jaccard"] = df["support"] / (
                df["antecedent support"] + df["consequent support"] - df["support"]
            )
        else:
            print("Cannot compute Jaccard (missing columns)")
            return None

    plt.figure(figsize=(4.5, 4.5))
    scatter = plt.scatter(
        df["lift"], df["jaccard"],
        s=df["confidence"] * 100,
        c=df["support"],
        cmap="viridis", alpha=0.7, edgecolors="k"
    )

    plt.colorbar(scatter, label="Support")
    plt.xlabel("Lift")
    plt.ylabel("Jaccard Similarity")
    plt.title(f"Lift vs Jaccard Similarity ({repo_name})")
    plt.grid(True)
    plt.tight_layout()

    out_file = os.path.join(out_dir, f"lift_vs_jaccard_{repo_name}.pdf")
    plt.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved bubble plot to {out_file}")
    return out_file


def plot_lhs_rhs_severity(repo_name, in_dir=".", out_dir=".", top_n=20):
    """
    Plot bubble chart of LHS groups vs RHS severity.
    """
    in_file = os.path.join(in_dir, f"rules_{repo_name}.csv")
    if not os.path.exists(in_file):
        print(f"Missing rules file: {in_file}")
        return None

    rules = pd.read_csv(in_file)
    if "Left_Hand_Side" not in rules.columns or "Right_Hand_Side" not in rules.columns:
        print(f"Missing expected columns in {in_file}")
        return None

    rules["LHS"] = rules["Left_Hand_Side"].str.strip()
    rules["RHS"] = rules["Right_Hand_Side"].str.strip()

    def classify(code):
        code = code.lower()
        if "fatal" in code or code.startswith("f"):
            return "Fatal"
        if "error" in code or code.startswith("e"):
            return "Error"
        if "warning" in code or code.startswith("w"):
            return "Warning"
        if "refactor" in code or code.startswith("r"):
            return "Refactor"
        if "convention" in code or code.startswith("c"):
            return "Convention"
        return "Other"

    severity_rank = {
        "Fatal": 0,
        "Error": 1,
        "Warning": 2,
        "Refactor": 3,
        "Convention": 4,
        "Other": 5,
    }

    def classify_group(lhs_string):
        codes = [c.strip() for c in lhs_string.split(",")]
        severities = [classify(c) for c in codes]
        return min(severities, key=lambda s: severity_rank[s])

    rules["LHS_Severity"] = rules["LHS"].apply(classify_group)
    rules["RHS_Severity"] = rules["RHS"].apply(classify)

    lhs_counts = rules["LHS"].value_counts()
    top_lhs = lhs_counts.head(top_n).index.tolist()
    rules_top = rules[rules["LHS"].isin(top_lhs)].copy()

    agg = rules_top.groupby(["LHS", "RHS_Severity"])["support"].sum().reset_index()
    agg["LHS_Severity"] = agg["LHS"].apply(classify_group)

    size_scale = 2000
    agg["size"] = agg["support"] * size_scale

    lhs_order = lhs_counts.head(top_n).index.tolist()
    rhs_order = ["Fatal", "Error", "Warning", "Refactor", "Convention", "Other"]
    agg["LHS"] = pd.Categorical(agg["LHS"], categories=lhs_order, ordered=True)
    agg["RHS_Severity"] = pd.Categorical(agg["RHS_Severity"], categories=rhs_order, ordered=True)

    _, ax = plt.subplots(figsize=(7.5, 4.0))
    # scatter = ax.scatter(
    #     x=agg["LHS"], y=agg["RHS_Severity"],
    #     s=agg["size"],
    #     c=agg["RHS_Severity"].cat.codes,
    #     cmap="Reds", alpha=0.8, edgecolors="k", linewidths=0.3
    # )

    ax.set_xlabel("LHS Rule Group")
    ax.set_ylabel("RHS Severity")
    ax.set_title(f"LHS vs RHS Severity ({repo_name})")
    ax.tick_params(axis="x", rotation=90)

    for support_val in [0.001, 0.005, 0.01]:
        ax.scatter([], [], s=support_val * size_scale, c="gray", alpha=0.5,
                   label=f"Support ≈ {support_val:.3f}")
    legend = ax.legend(title="Bubble Size", loc="upper right", fontsize=6, frameon=True)
    legend.get_title().set_fontsize(7)

    plt.tight_layout()
    out_file = os.path.join(out_dir, f"lhs_rhs_severity_{repo_name}.pdf")
    plt.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved severity bubble plot to {out_file}")
    return out_file


def plot_shared_rules_upset(repo_names, in_dir=".", out_dir="."):
    """
    Create an UpSet plot of shared rules with error consequents across multiple repos.
    """
    error_pattern = re.compile(r"^E\d{4}$")
    filtered = {}
    sets = {}

    for repo in repo_names:
        path = os.path.join(in_dir, f"rules_{repo}.csv")
        if not os.path.exists(path):
            print(f"Missing rules file: {path}")
            continue

        df = pd.read_csv(path)
        if "antecedents" not in df.columns or "consequents" not in df.columns:
            print(f"Missing antecedents/consequents in {path}")
            continue

        def safe_frozenset_parse(x):
            try:
                return frozenset(ast.literal_eval(x))
            except (ValueError, SyntaxError):
                return frozenset()

        df["antecedents"] = df["antecedents"].apply(safe_frozenset_parse)
        df["consequents"] = df["consequents"].apply(safe_frozenset_parse)

        df = df[
            df["antecedents"].apply(lambda x: len(x) <= 3) &
            df["consequents"].apply(lambda x: any(error_pattern.match(e) for e in x))
        ].copy()

        df = df.copy()
        df["rule_key"] = [
            (frozenset(a), frozenset(c))
            for a, c in zip(df["antecedents"], df["consequents"])
        ]
        filtered[repo] = df
        sets[repo] = set(df["rule_key"])

        print(f"{repo}: {len(df)} rules with error(s) in the consequent")

    if not sets:
        print("No repos to compare.")
        return None

    shared = set.intersection(*sets.values())
    print(f"Shared rules across all repos: {len(shared)}")

    if not shared:
        print("No shared rules to plot.")
        return None

    plot_data = {
        repo: {f"{tuple(a)} → {tuple(c)}" for (a, c) in rules}
        for repo, rules in sets.items()
    }

    upset_data = from_contents(plot_data)
    plt.figure(figsize=(7, 3.5))
    upset_plot(upset_data, orientation="horizontal", sort_by="cardinality")
    plt.title("Shared Rules with Error Consequents (≤3 Antecedents)")
    plt.subplots_adjust(top=0.9)

    out_file = os.path.join(out_dir, "shared_rules_upset_errors.pdf")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved UpSet plot to {out_file}")
    return out_file


def plot_shared_rules_upset_dynamic(sets, out_file):
    """
    Plot an upset plot of shared 1→1 rules across repos.

    Parameters
    ----------
    sets : dict[str, set]
        Mapping of repo name → set of rule strings.
    out_file : str or Path
        Path to save the figure (PDF).
    """
    upset_data = from_contents(sets)
    plt.figure(figsize=(7, 3.5))
    upset_plot(upset_data, orientation='horizontal', sort_by='cardinality')
    plt.title("Shared Association Rules (1→1 Match with Dynamic Threshold)")
    plt.tight_layout()
    plt.savefig(out_file, format='pdf', bbox_inches='tight')
    plt.close()
    return out_file


def extract_severity(code: str) -> str:
    """Map RHS error code prefix to severity class."""
    mapping = {
        "F": "fatal",
        "E": "error",
        "W": "warning",
        "R": "refactor",
        "C": "convention",
    }
    if not isinstance(code, str) or not code:
        return "other"
    first = code.split(",")[0].strip().split("-")[0].upper()
    return mapping.get(first[0], "other")


def plot_lhs_rhs_severity_bubble(repo_name: str, in_dir=".", out_dir="."):
    """
    Create a bubble plot of antecedent groups (LHS) vs. consequent severity classes.

    Parameters
    ----------
    repo_name : str
        Repository name. Expects file `rules_new_{repo_name}.csv` in in_dir.
    in_dir : str or Path
        Directory containing input CSV.
    out_dir : str or Path
        Directory where the PDF will be saved.

    Returns
    -------
    Path
        Path to the saved PDF file.
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file = in_dir / f"rules_new_{repo_name}.csv"
    if not file.exists():
        raise FileNotFoundError(f"Expected rules file not found: {file}")

    rules = pd.read_csv(file)
    rules["LHS"] = rules["Left_Hand_Side"]
    rules["RHS"] = rules["Right_Hand_Side"]

    # Map RHS → severity
    rules["RHS_Severity"] = rules["RHS"].apply(extract_severity)
    rules = rules[rules["RHS_Severity"] != "fatal"]

    # Normalize LHS
    rules["LHS_Group"] = rules["LHS"].apply(lambda x: x.split(",")[0].strip())

    # Aggregate
    agg = (
        rules.groupby(["LHS_Group", "RHS_Severity"])
        .agg(
            n_rhs_codes=("Right_Hand_Side", lambda x: len(set(x))),
            max_lift=("lift", "max"),
        )
        .reset_index()
    )

    # Top 20 LHS by spread
    top_lhs = agg.groupby("LHS_Group")["n_rhs_codes"].sum().nlargest(20).index
    plot_df = agg[agg["LHS_Group"].isin(top_lhs)].copy()

    # Axis mapping
    lhs_order = plot_df["LHS_Group"].value_counts().index.tolist()
    x_map = {lhs: i for i, lhs in enumerate(lhs_order)}
    y_order = ["convention", "refactor", "error", "warning"]  # fixed order
    y_map = {sev: i for i, sev in enumerate(y_order)}

    plot_df["x"] = plot_df["LHS_Group"].map(x_map)
    plot_df["y"] = plot_df["RHS_Severity"].map(y_map)
    plot_df["size"] = plot_df["n_rhs_codes"] * 10

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # Vertical dotted grid lines
    for x_val in x_map.values():
        ax.axvline(x=x_val, color="gray", linestyle=":", linewidth=0.5, zorder=0)

    norm = plt.Normalize(plot_df["max_lift"].min(), plot_df["max_lift"].max()) # type: ignore
    sc = ax.scatter(
        plot_df["x"],
        plot_df["y"],
        s=plot_df["size"],
        c=plot_df["max_lift"],
        cmap="Reds",
        norm=norm,
        edgecolors="k",
        alpha=0.8,
        zorder=10,
    )

    # Axes & labels
    ax.set_xticks(list(x_map.values()))
    ax.set_xticklabels(list(x_map.keys()), rotation=90)
    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(y_order)
    ax.set_xlabel("Antecedent Messages with the Most Consequent Messages")
    ax.set_ylabel("Consequent Severity Class")
    ax.set_title(
        "Antecedent Patterns Triggering Consequent Message Classes\n"
        "(Size = Count, Color = Max Lift)"
    )

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Max Lift")

    # Bubble size legend
    legend_sizes = [10, 30, 50]
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"{s} RHS codes",
            markerfacecolor="gray",
            markersize=np.sqrt(s * 10),
        )
        for s in legend_sizes
    ]
    ax.legend(
        handles=legend_handles,
        title="# RHS Codes",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=len(legend_sizes),
        frameon=False,
    )

    plt.tight_layout()

    out_file = out_dir / f"lhs_rhs_severity_bubble_{repo_name}.pdf"
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)

    return out_file


def plot_rule_network(repo_name, in_dir=".", out_dir="."):
    """
    Plot an association rule network graph for a given repository.

    Returns
    -------
    str
        Path to the saved PDF file.
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rules = pd.read_csv(in_dir / f"rules_new_{repo_name}.csv")

    # Build directed graph
    graph = nx.DiGraph()
    for _, row in rules.iterrows():
        lhs = row["Left_Hand_Side"]
        rhs = row["Right_Hand_Side"]
        graph.add_node(lhs)
        graph.add_node(rhs)
        graph.add_edge(lhs, rhs, lift=row["lift"], confidence=row["confidence"])

    # Node colors by out-degree
    out_degree_dict = dict(graph.out_degree()) # type: ignore
    node_colors = list(out_degree_dict.values())
    norm = plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)) # type: ignore
    cmap = matplotlib.colormaps['YlGnBu']

    fig, ax = plt.subplots(figsize=(7, 3.5))
    pos = nx.spring_layout(graph, k=0.4, seed=42)

    nx.draw_networkx_edges(graph, pos, alpha=0.3, arrows=True, arrowstyle="-|>", ax=ax)

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=80,
        node_color=node_colors, # type: ignore
        cmap=cmap,
        edgecolors="black",
        linewidths=0.5,
        ax=ax,
    )

    # Add labels with white halo
    for node, (x, y) in pos.items():
        ax.text(
            x,
            y,
            node,
            fontsize=4,
            ha="center",
            va="center",
            color="black",
            path_effects=[pe.withStroke(linewidth=1.2, foreground="white")],
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Out-degree (Number of Outgoing Rules)", fontsize=6)

    ax.set_title(f"Association Rule Graph — {repo_name}", fontsize=8)
    ax.axis("off")

    out_file = out_dir / f"assoc_network_outdegree_{repo_name}.pdf"
    plt.tight_layout()
    plt.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return str(out_file)


def plot_asymmetry_matrix(repo_name, in_dir=".", out_dir="."):
    """
    Plot lower-triangular confidence asymmetry matrix for 1→1 rules.

    Returns
    -------
    Path or None
        Path to saved PDF, or None if no valid asymmetry rules.
    """
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    csv_file = in_dir / f"rules_{repo_name}.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")

    rules = pd.read_csv(csv_file)

    # Keep only 1→1 rules
    rules = rules[
        (rules['antecedents'].str.count(',') == 0) &
        (rules['consequents'].str.count(',') == 0)
    ].copy()

    if rules.empty:
        print(f"No 1→1 rules for {repo_name}")
        return None

    # Normalize A,B
    rules['A'] = rules['Left_Hand_Side'].str.strip()
    rules['B'] = rules['Right_Hand_Side'].str.strip()
    rules['key'] = rules['A'] + '→' + rules['B']
    rules['reverse_key'] = rules['B'] + '→' + rules['A']

    # Lookup reverse confidence
    reverse = rules[['key', 'confidence']].rename(
        columns={'key': 'reverse_key', 'confidence': 'confidence_rev'}
    )
    merged = rules.merge(reverse, on='reverse_key')
    merged['asymmetry'] = merged['confidence'] - merged['confidence_rev']

    if merged.empty:
        print(f"No asymmetric pairs for {repo_name}")
        return None

    # Pivot to heatmap matrix
    heatmap_data = merged.pivot(index='A', columns='B', values='asymmetry')

    # Mask upper triangle
    mask = np.triu(np.ones_like(heatmap_data, dtype=bool))

    # Style
    matplotlib.rcParams.update({
        'font.size': 6,
        'axes.labelsize': 6,
        'axes.titlesize': 7,
        'legend.fontsize': 6,
        'xtick.labelsize': 5,
        'ytick.labelsize': 5,
        'figure.figsize': (7, 7),
        'savefig.dpi': 300,
        'pdf.fonttype': 42,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial']
    })

    # Plot
    plt.figure(figsize=(7, 7))
    ax = sns.heatmap(
        heatmap_data,
        mask=mask,
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 5},
        cbar_kws={"label": "Confidence Asymmetry"}
    )

    # Add subtle gray grid to lower triangle
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            if i >= j:
                ax.add_patch(plt.Rectangle( # type: ignore
                    (j, i), 1, 1,
                    fill=False,
                    edgecolor='lightgray',
                    linewidth=0.5
                ))

    plt.xticks(rotation=90, ha="right", fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.title(f"Confidence Asymmetry Matrix ({repo_name})", fontsize=8)
    plt.xlabel("Consequent Warning Code (B)")
    plt.ylabel("Antecedent Warning Code (A)")

    out_file = out_dir / f"confidence_asymmetry_{repo_name}.pdf"
    plt.tight_layout()
    plt.savefig(out_file, format="pdf", bbox_inches="tight")
    plt.close()

    return out_file
