# Uncovering Patterns in Python Static Code Warnings Using Association Rule Mining

This repository contains the implementation accompanying the submission:

> *Uncovering Patterns in Python Static Code Warnings Using Association Rule Mining*  
> Submitted to ICSE 2026 (Double-Anonymized Review)

---

## Overview
The code in this repository processes static analysis outputs (e.g., from **Pylint**) and applies **Association Rule Mining** (ARM) to discover co-occurrence patterns between warnings.

It includes:
- Parsing of Pylint warning reports (JSON/CSV)
- Transformation into transactional datasets
- Mining of association rules using the **Apriori algorithm**
- Computation of rule metrics (support, confidence, lift, Jaccard)
- Basic visualization scripts (bubble plots, heatmaps, rule networks)

---

## Usage
1. Run Pylint on a Python repository and export results as JSON/CSV.
2. Place the reports in the `data/` folder.
3. Use the provided preprocessing and mining scripts to generate association rules.
4. Visualizations can be created using the plotting utilities.

