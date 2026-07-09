#!/usr/bin/env python3
"""
GSEA (prerank) on PyDESeq2 flat result CSVs in this folder, via gseapy.

Threshold-free: ALL genes are ranked by log2FoldChange (per user's choice),
then prerank GSEA is run against 5 MSigDB/Enrichr libraries.
(No pvalue or LFC cutoff -- GSEA uses the full ranked list by design.)

Reads flat files named:  {celltype}__group_age_z_sex_apoE__{contrast}.csv
Outputs go to:           GSEA_prerank_wald/
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
OUT = BASE / "GSEA_prerank_wald"
DESIGN_TOKEN = "__group_age_z_sex_apoE__"

RANK_COL = "stat"           # Wald statistic (user-chosen for this run)
PERMUTATION_NUM = 1000
MIN_SIZE = 10
MAX_SIZE = 500
SEED = 42
SLEEP_BETWEEN_JOBS_SEC = 2

# gene_sets passed to gp.prerank as Enrichr library names (gseapy downloads/caches them)
GENE_SET_LIBS = [
    ("MSigDB_Hallmark_2020", "Hallmark"),
    ("KEGG_2021_Human", "KEGG"),
    ("Reactome_2022", "Reactome"),
    ("WikiPathway_2023_Human", "WikiPathways"),
    ("GO_Biological_Process_2023", "GO_BP"),
]


def contrast_files() -> list[Path]:
    return sorted(p for p in BASE.glob(f"*{DESIGN_TOKEN}*.csv") if p.is_file())


def parse_name(path: Path) -> tuple[str, str]:
    celltype, contrast = path.stem.split(DESIGN_TOKEN, 1)
    return celltype, contrast


def ranked_genes(df: pd.DataFrame) -> pd.Series:
    sub = df[["gene", RANK_COL]].dropna().copy()
    sub["gene"] = sub["gene"].astype(str).str.strip().str.upper()
    sub[RANK_COL] = pd.to_numeric(sub[RANK_COL], errors="coerce")
    sub = sub.dropna(subset=[RANK_COL])
    rnk = sub.groupby("gene")[RANK_COL].mean().sort_values(ascending=False)
    return rnk


def barplot_nes(df: pd.DataFrame, out_png: Path, title: str) -> None:
    if df.empty:
        return
    n_plot = min(30, len(df))
    plot_df = df.head(n_plot).copy()
    padj_col = "padj" if "padj" in plot_df.columns else ("pval" if "pval" in plot_df.columns else None)
    fig, ax = plt.subplots(figsize=(10, max(6, n_plot * 0.25)))
    y_pos = range(len(plot_df))
    if padj_col is not None:
        colors = ["#e41a1c" if pd.to_numeric(plot_df[padj_col].iloc[i], errors="coerce") < 0.05
                  else "#377eb8" for i in range(len(plot_df))]
    else:
        colors = "#377eb8"
    ax.barh(list(y_pos), pd.to_numeric(plot_df["NES"], errors="coerce").values, color=colors)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(plot_df["pathway"].tolist(), fontsize=7)
    ax.set_xlabel("NES  (red = padj < 0.05)")
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def run_gsea(rnk: pd.Series, label: str) -> int:
    tables = OUT / "tables"
    plots = OUT / "plots"
    tables.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for lib, short in GENE_SET_LIBS:
        out_table = tables / f"{label}_{short}_gsea.csv"
        out_plot = plots / f"{label}_{short}_NESbar.png"
        if out_table.is_file():
            n_ok += 1
            continue
        try:
            pr = gp.prerank(
                rnk=rnk,
                gene_sets=lib,
                outdir=None,
                permutation_num=PERMUTATION_NUM,
                no_plot=True,
                seed=SEED,
                min_size=MIN_SIZE,
                max_size=MAX_SIZE,
                verbose=False,
            )
            df = pr.res2d.copy()
            if "Term" in df.columns:
                df["pathway"] = df["Term"].astype(str)
                df = df.drop(columns=["Term"], errors="ignore")
            if "NOM p-val" in df.columns and "pval" not in df.columns:
                df = df.rename(columns={"NOM p-val": "pval"})
            if "FDR q-val" in df.columns and "padj" not in df.columns:
                df = df.rename(columns={"FDR q-val": "padj"})
            df["NES"] = pd.to_numeric(df["NES"], errors="coerce")
            df = df.sort_values("NES", ascending=False)
            df.to_csv(out_table, index=False)
            barplot_nes(df, out_plot, f"{label} - {short}")
            n_ok += 1
        except Exception as e:
            (OUT / "errors.log").open("a").write(f"{label} {lib}: {e}\n")
    return n_ok


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = contrast_files()
    print(f"Found {len(files)} contrast CSVs. Ranking by {RANK_COL}.", flush=True)

    log_rows = []
    errors = []
    done = 0

    for csv_path in files:
        celltype, contrast = parse_name(csv_path)
        label = f"{celltype}__{contrast}"
        try:
            df = pd.read_csv(csv_path)
            rnk = ranked_genes(df)
            n_ok = run_gsea(rnk, label)
            log_rows.append({
                "celltype": celltype,
                "contrast": contrast,
                "n_ranked_genes": int(len(rnk)),
                "libraries_ok": n_ok,
            })
            print(f"{label}: {len(rnk)} ranked genes -> {n_ok}/{len(GENE_SET_LIBS)} libs", flush=True)
            done += 1
            pd.DataFrame(log_rows).to_csv(OUT / "SUMMARY_gsea_prerank_wald_log.csv", index=False)
        except Exception:
            errors.append(f"{csv_path}:\n{traceback.format_exc()}")
            print(f"  ERROR on {label}", flush=True)
        gc.collect()
        time.sleep(SLEEP_BETWEEN_JOBS_SEC)

    pd.DataFrame(log_rows).to_csv(OUT / "SUMMARY_gsea_prerank_wald_log.csv", index=False)
    if errors:
        (OUT / "SUMMARY_gsea_prerank_wald_errors.txt").write_text("\n\n".join(errors))
    print(f"\nDONE. Processed: {done}, errors: {len(errors)}")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
