#!/usr/bin/env python3
"""
ORA (Enrichr) on PyDESeq2 flat CSVs, split by direction.

Gene selection: pvalue < 0.05, then split by sign of log2FoldChange:
    UP   = pvalue < 0.05 AND log2FoldChange > 0
    DOWN = pvalue < 0.05 AND log2FoldChange < 0
NO magnitude/LFC threshold (only the sign is used to split up vs down).

Reads flat files named:  {celltype}__group_age_z_sex_apoE__{contrast}.csv
Outputs go to:           ORA_pval005_updown/
"""

from __future__ import annotations

import gc
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import gseapy as gp
except ImportError:
    print("ERROR: pip install gseapy")
    sys.exit(1)

BASE = Path(__file__).resolve().parent
OUT = BASE / "ORA_pval005_updown"
DESIGN_TOKEN = "__group_age_z_sex_apoE__"

P_thr = 0.05
MIN_GENES = 5
ADJ_PVAL = 0.2
TOP_N = 25
SLEEP_BETWEEN_JOBS_SEC = 2

ENRICHR_LIBRARIES = [
    "KEGG_2021_Human",
    "Reactome_2022",
    "MSigDB_Hallmark_2020",
    "WikiPathway_2023_Human",
    "GO_Biological_Process_2023",
]
PLOT_LIBRARIES = [
    ("WikiPathway_2023_Human", "WikiPathways"),
    ("MSigDB_Hallmark_2020", "Hallmark"),
    ("GO_Biological_Process_2023", "GO_BP"),
    ("Reactome_2022", "Reactome"),
    ("KEGG_2021_Human", "KEGG"),
]


def contrast_files() -> list[Path]:
    return sorted(p for p in BASE.glob(f"*{DESIGN_TOKEN}*.csv") if p.is_file())


def parse_name(path: Path) -> tuple[str, str]:
    celltype, contrast = path.stem.split(DESIGN_TOKEN, 1)
    return celltype, contrast


def filter_direction(df: pd.DataFrame, direction: str) -> list[str]:
    sub = df.dropna(subset=["pvalue", "log2FoldChange", "gene"]).copy()
    sub["pvalue"] = pd.to_numeric(sub["pvalue"], errors="coerce")
    sub["log2FoldChange"] = pd.to_numeric(sub["log2FoldChange"], errors="coerce")
    sub = sub[sub["pvalue"] < P_thr]
    if direction == "up":
        sub = sub[sub["log2FoldChange"] > 0]
    elif direction == "down":
        sub = sub[sub["log2FoldChange"] < 0]
    else:
        raise ValueError(direction)
    genes = sub["gene"].astype(str).str.strip().str.upper().unique().tolist()
    return sorted(genes)


def plot_enrichr(csv_path: Path, barplots_dir: Path, base_name: str) -> None:
    barplots_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    pcol = "Adjusted P-value" if "Adjusted P-value" in df.columns else "P-value"
    if pcol not in df.columns:
        return
    for gene_set_value, short_name in PLOT_LIBRARIES:
        sub = df[(df["Gene_set"] == gene_set_value) & (df[pcol] < ADJ_PVAL)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(pcol).head(TOP_N)
        pvals = np.clip(sub[pcol].values, 1e-300, 1.0)
        y = -np.log10(pvals)
        terms_short = [t[:60] + "..." if len(t) > 60 else t for t in sub["Term"].tolist()]
        n = len(terms_short)
        fig, ax = plt.subplots(figsize=(8, max(5, n * 0.35)))
        y_pos = np.arange(n)[::-1]
        ax.barh(y_pos, y, color="steelblue", edgecolor="navy", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(terms_short, fontsize=8)
        ax.set_xlabel("-log10(Adjusted P-value)")
        ax.set_title(f"{base_name} — {short_name}")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(barplots_dir / f"{base_name}_{short_name}_barplot.png", dpi=150, bbox_inches="tight")
        plt.close()


def run_ora(genes: list[str], label: str) -> bool:
    if len(genes) < MIN_GENES:
        return False
    ora_tables = OUT / "ORA" / "tables"
    ora_barplots = OUT / "ORA" / "barplots"
    ora_tables.mkdir(parents=True, exist_ok=True)
    out_csv = ora_tables / f"{label}_enrichr.csv"
    if out_csv.is_file():
        return True
    all_dfs = []
    for lib in ENRICHR_LIBRARIES:
        try:
            enr = gp.enrichr(genes, gene_sets=lib, organism="human", outdir=None, no_plot=True, verbose=False)
            res = enr.results.copy()
            res["Gene_set"] = lib
            all_dfs.append(res)
        except Exception as e:
            (OUT / "ORA" / "errors.log").open("a").write(f"{label} {lib}: {e}\n")
    if not all_dfs:
        return False
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(out_csv, index=False)
    plot_enrichr(out_csv, ora_barplots, label)
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    deg_dir = OUT / "DEG_lists"
    deg_dir.mkdir(parents=True, exist_ok=True)
    files = contrast_files()
    print(f"Found {len(files)} contrast CSVs.", flush=True)

    log_rows = []
    errors = []
    for csv_path in files:
        celltype, contrast = parse_name(csv_path)
        label = f"{celltype}__{contrast}"
        try:
            df = pd.read_csv(csv_path)
            row = {"celltype": celltype, "contrast": contrast}
            for direction in ("up", "down"):
                genes = filter_direction(df, direction)
                (deg_dir / f"{label}__pval005_{direction}_genes.txt").write_text(
                    "\n".join(genes) + ("\n" if genes else "")
                )
                ok = run_ora(genes, f"{label}__ORA_pval005_{direction}")
                row[f"n_{direction}"] = len(genes)
                row[f"ora_{direction}"] = ok
                print(f"{label} [{direction}]: {len(genes)} genes -> {'run' if ok else 'skip(<%d)'%MIN_GENES}", flush=True)
            log_rows.append(row)
            pd.DataFrame(log_rows).to_csv(OUT / "SUMMARY_ora_pval005_updown_log.csv", index=False)
        except Exception:
            errors.append(f"{csv_path}:\n{traceback.format_exc()}")
            print(f"  ERROR on {label}", flush=True)
        gc.collect()
        time.sleep(SLEEP_BETWEEN_JOBS_SEC)

    pd.DataFrame(log_rows).to_csv(OUT / "SUMMARY_ora_pval005_updown_log.csv", index=False)
    if errors:
        (OUT / "SUMMARY_ora_pval005_updown_errors.txt").write_text("\n\n".join(errors))
    print(f"\nDONE. Output: {OUT}")


if __name__ == "__main__":
    main()
