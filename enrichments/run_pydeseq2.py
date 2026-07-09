#!/usr/bin/env python3
"""
Pseudobulk differential expression aligned with DESEq2.v5.r
  - Input: AnnData (.h5ad) with counts layer (or .X), obs: group, sample_id, subcelltype, age, sex, apoE.
  - PyDESeq2: per-subcelltype × design × contrast (Healthy / Dyslexia / AD).
  - decoupler (optional): ULM on SIGNOR pathway→gene network using log1p pseudobulk (group_only runs).
  - Optional apeGLM-style LFC shrink via PyDESeq2 (``DESEQ_LFC_SHRINK``); not the same as R ``lfcShrink(type="ashr")``.
  - Optional pseudobulk PCA scatter (``DESEQ_PCA_SANITY``): log1p(counts), PC1 vs PC2, colored by ``group``.

Install:
  pip install pydeseq2 decoupler anndata scanpy pandas numpy scipy
  # for PCA plot: pip install scikit-learn matplotlib

Docs: https://pydeseq2.readthedocs.io/  https://decoupler.readthedocs.io/
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Config (mirror DESEq2.v5.r)
# -----------------------------------------------------------------------------
INPUT_H5AD = "18_merged_AG_040526.27.celltypes.balanced_to_54331_cells_complete.h5ad"
GROUP_COL = "group"
CELLTYPE_COL = "subcelltype"
REPLICATE_COL = "sample_id"
COUNTS_LAYER = "counts"
MIN_CELLS_PER_SAMPLE = 10
MIN_TOTAL_CELLS_PER_CELLTYPE = 50
MIN_REPS_PER_GROUP = 2
# At least two groups with MIN_REPS_PER_GROUP samples each (DESEq2.v5.r uses literal 4 here when min group size is 2).
MIN_SAMPLES_STAGE = MIN_REPS_PER_GROUP * 2
PADJ_CUTOFF = 0.05
OUT_DIR = Path("DE_results_v5_multi_design_python") / "_v5"

GENE_MIN_COUNT = 10
GENE_MIN_SAMPLES = 3

COMPARISONS: dict[str, tuple[str, str, str]] = {
    "AD_vs_Healthy": ("group", "AD", "Healthy"),
    "Dyslexia_vs_Healthy": ("group", "Dyslexia", "Healthy"),
    "Dyslexia_vs_AD": ("group", "Dyslexia", "AD"),
}

DESIGN_SPECS: dict[str, dict[str, Any]] = {
    "group_only": {"formula": "~ group", "covars": []},
    "group_age_z": {"formula": "~ group + age_z", "covars": ["age_z"]},
    "group_sex": {"formula": "~ group + sex", "covars": ["sex"]},
    "group_age_z_sex": {"formula": "~ group + age_z + sex", "covars": ["age_z", "sex"]},
    "group_apoE": {"formula": "~ group + apoE", "covars": ["apoE"]},
    "group_age_z_sex_apoE": {"formula": "~ group + age_z + sex + apoE", "covars": ["age_z", "sex", "apoE"]},
}

INFERENCE_CPUS = int(os.environ.get("PYDESEQ_INFERENCE_CPUS", "1"))
RUN_DECOUPLER = os.environ.get("DESEQ_RUN_DECOUPLER", "1").lower() in ("1", "true", "yes")
# PyDESeq2 apeGLM-style LFC shrink (not R DESeq2::lfcShrink type="apeglm" nor ashr).
# Skipped when the Wald contrast is composite (several design columns), e.g. some non-ref vs non-ref pairs.
RUN_LFC_SHRINK = os.environ.get("DESEQ_LFC_SHRINK", "1").lower() in ("1", "true", "yes")
RUN_PCA_SANITY = os.environ.get("DESEQ_PCA_SANITY", "0").lower() in ("1", "true", "yes")


def safe_slug(x: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9]+", "_", str(x))


def pseudobulk_pca_plot(
    cnt_sxg: pd.DataFrame,
    meta_loc: pd.DataFrame,
    out_png: Path,
    title_suffix: str = "",
) -> None:
    """PCA on log1p pseudobulk matrix (samples × genes); scatter colored by ``GROUP_COL``."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError as e:
        print(f"  PCA sanity skipped (need scikit-learn, matplotlib): {e}")
        return

    ix = [str(x) for x in cnt_sxg.index]
    g = meta_loc.reindex(ix)[GROUP_COL].astype(str)
    X = np.log1p(np.maximum(cnt_sxg.values.astype(np.float64), 0.0))
    n_s, n_f = X.shape
    if n_s < 3 or n_f < 2:
        return
    k = min(2, n_s - 1, n_f)
    if k < 1:
        return
    coords = PCA(n_components=k).fit_transform(X)
    if k == 1:
        coords = np.column_stack([coords.ravel(), np.zeros(n_s)])

    palette = {"Healthy": "#2ca02c", "Dyslexia": "#1f77b4", "AD": "#d62728"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for lab in sorted(g.unique()):
        m = g.values == lab
        if not m.any():
            continue
        c = palette.get(lab, "#7f7f7f")
        ax.scatter(
            coords[m, 0],
            coords[m, 1],
            c=c,
            label=lab,
            alpha=0.88,
            edgecolors="k",
            linewidths=0.35,
            s=42,
        )
    ax.set_xlabel("PC1 (log1p pseudobulk)")
    ax.set_ylabel("PC2 (log1p pseudobulk)")
    ax.set_title(f"Pseudobulk PCA{title_suffix}")
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = float(s.mean(skipna=True))
    sd = float(s.std(skipna=True, ddof=1))
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


def load_x_cells_x_genes(adata):
    import scipy.sparse as sp

    if COUNTS_LAYER in adata.layers:
        X = adata.layers[COUNTS_LAYER]
    else:
        X = adata.X
    if sp.issparse(X):
        return X
    return np.asarray(X, dtype=np.float64)


def pseudobulk_one(adata, ct: str) -> pd.DataFrame | None:
    """genes × samples integer counts."""
    import scipy.sparse as sp

    obs = adata.obs
    genes = pd.Index(adata.var_names.astype(str))
    cell_mask = (obs[CELLTYPE_COL] == ct).to_numpy()
    if int(cell_mask.sum()) < MIN_TOTAL_CELLS_PER_CELLTYPE:
        return None

    X = load_x_cells_x_genes(adata)
    X = X[cell_mask, :]
    sids_raw = obs.loc[obs[CELLTYPE_COL] == ct, REPLICATE_COL].astype(str)

    vc = sids_raw.value_counts()
    good_samples = vc[vc >= MIN_CELLS_PER_SAMPLE].index
    if good_samples.size < MIN_SAMPLES_STAGE:
        return None

    keep = sids_raw.isin(good_samples).to_numpy()
    X = X[keep, :]
    sids = sids_raw[keep]

    uniq = [str(x) for x in good_samples]
    sums = []
    if sp.issparse(X):
        X = X.tocsr()
        for sid in uniq:
            idx = np.where(sids.values == sid)[0]
            sums.append(np.asarray(X[idx, :].sum(axis=0)).ravel())
    else:
        for sid in uniq:
            idx = np.where(sids.values == sid)[0]
            sums.append(np.asarray(X[idx, :]).sum(axis=0).ravel())

    mat = np.column_stack([np.asarray(s, dtype=np.int64) for s in sums])
    return pd.DataFrame(mat, index=genes, columns=uniq)


def build_sample_meta(adata_obs: pd.DataFrame) -> pd.DataFrame:
    cols_req = [REPLICATE_COL, GROUP_COL, "sex", "age", "apoE"]
    meta = adata_obs[cols_req].drop_duplicates(subset=[REPLICATE_COL], keep="first").copy()
    dup = meta[REPLICATE_COL].duplicated()
    if dup.any():
        raise ValueError(f"Duplicate {REPLICATE_COL} rows in metadata.")

    meta = meta.set_index(REPLICATE_COL, drop=True)
    meta.index = meta.index.astype(str)
    meta.index.name = REPLICATE_COL
    meta[GROUP_COL] = pd.Categorical(meta[GROUP_COL], categories=["Healthy", "Dyslexia", "AD"], ordered=True)
    meta["sex"] = meta["sex"].astype(str).astype("category")
    meta["apoE"] = meta["apoE"].astype(str).astype("category")
    meta["age_z"] = zscore(meta["age"])
    return meta


def filter_genes_ddstyle(counts_sxg: pd.DataFrame) -> pd.DataFrame:
    """samples × genes."""
    ok = ((counts_sxg >= GENE_MIN_COUNT).sum(axis=0)) >= GENE_MIN_SAMPLES
    genes = counts_sxg.columns[ok.fillna(False).to_numpy()]
    return counts_sxg[genes].copy()


def meta_columns_for_design(needed_covars: list[str]) -> list[str]:
    """Columns passed to PyDESeq2: group + design covariates only (complete cases on this set).

    Note: DESEq2.v5.r also appends ``sex`` to colData for every design; we omit it when not in
    ``needed_covars`` so missing sex does not drop samples from e.g. ``group_only`` analyses.
    """
    return list(dict.fromkeys([GROUP_COL, *needed_covars]))


def _lfc_coeff_for_apeglm_shrink(ds) -> str | None:
    """Single design column matching the contrast (PyDESeq2 ``lfc_shrink`` needs one coeff name)."""
    cv = np.asarray(ds.contrast_vector, dtype=float).ravel()
    cols = ds.LFC.columns
    if len(cv) != len(cols):
        return None
    idx = [i for i in range(len(cv)) if abs(cv[i]) > 1e-10]
    if len(idx) != 1:
        return None
    return str(cols[idx[0]])


def check_factor_levels(meta: pd.DataFrame, needed_covars: list[str]) -> str | None:
    """Covariates omitted from formula (e.g. sex in R colData-only) are not validated here."""
    if "sex" in needed_covars and meta["sex"].astype(str).dropna().nunique() < 2:
        return "sex has <2 levels"
    if "apoE" in needed_covars and meta["apoE"].astype(str).dropna().nunique() < 2:
        return "apoE has <2 levels"
    if "age_z" in needed_covars and meta["age_z"].dropna().nunique() < 2:
        return "age_z has <2 unique values"
    return None


def run_pydeseq2_contrasts(
    counts_sxg: pd.DataFrame,
    meta_full: pd.DataFrame,
    design_id: str,
    formula_str: str,
    needed_covars: list[str],
) -> dict[str, Any] | None:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    from pydeseq2.ds import DeseqStats

    cols = [c for c in meta_columns_for_design(needed_covars) if c in meta_full.columns]
    samples = counts_sxg.index.astype(str)

    meta = meta_full.reindex(samples)
    meta = meta[cols]

    cc = meta.notna().all(axis=1)
    if (~cc).any():
        meta = meta.loc[cc].copy()
        counts_sxg = counts_sxg.loc[meta.index.astype(str)]

    fl = check_factor_levels(meta, needed_covars)
    if fl is not None:
        print(f"  [{design_id}] skip: {fl}")
        return None

    if counts_sxg.shape[0] < MIN_SAMPLES_STAGE:
        print(f"  [{design_id}] skip: too few complete-case samples")
        return None

    counts_tf = counts_sxg.copy()
    counts_tf[counts_tf < 0] = 0
    counts_tf = np.round(counts_tf).astype(np.int32)
    counts_tf = filter_genes_ddstyle(counts_tf.astype(np.int64))
    if counts_tf.shape[1] == 0:
        print(f"  [{design_id}] skip: no genes after filtering")
        return None

    inference = DefaultInference(n_cpus=max(1, INFERENCE_CPUS))
    try:
        dds = DeseqDataSet(
            counts=counts_tf,
            metadata=meta,
            design=formula_str,
            refit_cooks=True,
            inference=inference,
        )
        dds.deseq2()
    except Exception as e:
        print(f"  DESeq fit failed ({design_id}): {e}")
        return None

    inference2 = DefaultInference(n_cpus=max(1, INFERENCE_CPUS))
    result_list: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []

    def grp_counts(mm: pd.DataFrame, g1: str, g2: str) -> tuple[int, int]:
        vc = mm[GROUP_COL].astype(str).value_counts()
        a = int(vc[g1]) if g1 in vc.index else 0
        b = int(vc[g2]) if g2 in vc.index else 0
        return a, b

    for cmp_name, co_t in COMPARISONS.items():
        fac, lvl_num, lvl_den = co_t
        n_num, n_den = grp_counts(meta, lvl_num, lvl_den)
        if n_num < MIN_REPS_PER_GROUP or n_den < MIN_REPS_PER_GROUP:
            summary_rows.append(
                {
                    "design_id": design_id,
                    "comparison": cmp_name,
                    "group_num": lvl_num,
                    "group_den": lvl_den,
                    "n_samples_num": n_num,
                    "n_samples_den": n_den,
                    "status": "skipped_insufficient_reps",
                }
            )
            continue

        try:
            ds = DeseqStats(
                dds, contrast=[fac, lvl_num, lvl_den], inference=inference2, quiet=True
            )
            ds.summary()
            lfc_shrink_note = ""
            if RUN_LFC_SHRINK:
                coeff_sc = _lfc_coeff_for_apeglm_shrink(ds)
                if coeff_sc is None:
                    lfc_shrink_note = "skipped_composite_or_ambiguous_contrast"
                else:
                    try:
                        ds.lfc_shrink(coeff=coeff_sc)
                        lfc_shrink_note = f"apeglm:{coeff_sc}"
                    except Exception as se:
                        print(f"  LFC shrink failed ({design_id}, {cmp_name}): {se}")
                        lfc_shrink_note = f"error:{se}"
            else:
                lfc_shrink_note = "disabled"
        except Exception as e:
            print(f"  Wald test failed ({design_id}, {cmp_name}): {e}")
            summary_rows.append({"design_id": design_id, "comparison": cmp_name, "status": f"wald_error:{e}"})
            continue

        res_df = ds.results_df.copy()
        res_df.insert(0, "gene", res_df.index.astype(str))
        res_df["comparison"] = cmp_name
        res_df["design_id"] = design_id
        res_df["lfc_shrink"] = lfc_shrink_note

        padj = pd.to_numeric(res_df["padj"], errors="coerce")
        sig = padj.notna() & (padj < PADJ_CUTOFF)
        lf = pd.to_numeric(res_df["log2FoldChange"], errors="coerce")

        summary_rows.append(
            {
                "design_id": design_id,
                "comparison": cmp_name,
                "group_num": lvl_num,
                "group_den": lvl_den,
                "n_samples_num": n_num,
                "n_samples_den": n_den,
                "n_genes_tested": int(res_df.shape[0]),
                "n_deg_padj": int(sig.sum()),
                "n_up_in_num": int((sig & (lf > 0)).sum()),
                "n_up_in_den": int((sig & (lf < 0)).sum()),
                "n_up_in_num_lfc025": int((sig & (lf > 0.25)).sum()),
                "n_up_in_den_lfc025": int((sig & (lf < -0.25)).sum()),
                "status": "ok",
            }
        )
        result_list[cmp_name] = res_df

    return {"dds": dds, "results": result_list, "summary": pd.DataFrame(summary_rows)}


def omnipath_to_source_target(net: pd.DataFrame) -> pd.DataFrame | None:
    """Omnipath ``resource()`` tables are often wide; ULM expects long edges with ``source``, ``target`` [, ``weight``]."""
    if {"source", "target"}.issubset(net.columns):
        nt = net[["source", "target"]].copy()
        if "weight" in net.columns:
            nt["weight"] = pd.to_numeric(net["weight"], errors="coerce").fillna(1.0)
        else:
            nt["weight"] = 1.0
        return nt.drop_duplicates(["source", "target"])
    if "genesymbol" not in net.columns:
        return None
    id_cols = [c for c in ("genesymbol", "record_id") if c in net.columns]
    if len([c for c in net.columns if c not in id_cols]) == 0:
        return None
    m = net.melt(id_vars=id_cols, var_name="source", value_name="_membership").dropna(subset=["_membership"])
    m = m[m["_membership"].astype(str).str.len() > 0]
    m = m.rename(columns={"genesymbol": "target"})
    ed = m[["source", "target"]].assign(weight=1.0).drop_duplicates(["source", "target"])
    return ed if len(ed) > 12 else None


def run_decoupler_signor_ulm(
    adata_ct, pseudobulk_gene_x_sample: pd.DataFrame, _diag_dir: Path, ct_slug: str
) -> None:
    """ULM pathway scores (decoupler `dc.mt.ulm`): tries SIGNOR, then hallmark gene sets."""
    try:
        import scanpy as sc
        import decoupler as dc
    except Exception as e:
        print(f"  decoupler skip-import: {e}")
        return

    samples_all = [str(x) for x in pseudobulk_gene_x_sample.columns]
    pb_txs = pseudobulk_gene_x_sample.T.loc[samples_all].copy()

    meta_agg = adata_ct.obs.groupby(REPLICATE_COL, observed=False)[GROUP_COL].first().astype(str).reindex(samples_all)
    meta_agg = meta_agg.dropna()
    keep_ix = meta_agg.index.astype(str)
    pb_kept = pb_txs.loc[keep_ix]
    x = np.log1p(np.maximum(pb_kept.values.astype(np.float64), 0))

    obs_df = pd.DataFrame({GROUP_COL: meta_agg.values}, index=keep_ix.astype(str))
    gene_names = pb_kept.columns.astype(str)
    var_df = pd.DataFrame(index=gene_names)

    ad_pb = sc.AnnData(X=x, obs=obs_df, var=var_df)
    net: pd.DataFrame | None = None
    tag = ""

    try:
        net_raw = dc.op.resource("SIGNOR", verbose=False)
        tag = "SIGNOR"
        net = omnipath_to_source_target(net_raw)
        if net is None:
            raise ValueError("could not coerce SIGNOR to source/target")
    except Exception as e_s:
        print(f"  SIGNOR network skipped ({e_s}); hallmark fallback.")
        try:
            tag = "Hallmark"
            net = omnipath_to_source_target(dc.op.hallmark(verbose=False))
        except Exception as e_h:
            print(f"  decoupler hallmark skip: {e_h}")
            return

    tgt = pd.Index(net["target"].astype(str)).unique()
    genes = pd.Index(ad_pb.var_names.astype(str))
    ov = np.intersect1d(tgt, genes, assume_unique=False)
    if ov.size < 12:
        print("  pathway/target overlap too small; skip decoupler.")
        return

    out_dir = OUT_DIR / f"decoupler_ULM_{tag.replace(' ', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        dc.mt.ulm(ad_pb, net, source="source", target="target", weight="weight", verbose=False)
    except Exception as e:
        print(f"  dc.mt.ulm failed: {e}")
        return

    exported: list[str] = []
    for kk in sorted(ad_pb.obsm.keys()):
        if "ulm" not in kk.lower():
            continue
        slot = ad_pb.obsm[kk]
        arr = getattr(slot, "values", slot)
        out = pd.DataFrame(np.asarray(arr), index=ad_pb.obs_names.astype(str))
        cols = getattr(slot, "columns", None)
        if cols is not None:
            out.columns = cols.astype(str)
        path = out_dir / f"{ct_slug}__{kk}.csv"
        out.to_csv(path)
        exported.append(kk)
    if not exported:
        pd.DataFrame({"keys": list(ad_pb.obsm.keys())}).to_csv(out_dir / f"{ct_slug}_decoupler_debug_obsm.csv")


def main() -> None:
    import warnings

    import anndata as ad

    warnings.filterwarnings("ignore", category=UserWarning)

    # Jupyter/notebook paste: __file__ is undefined — fall back to the working directory.
    try:
        root = Path(__file__).resolve().parent
    except NameError:
        root = Path.cwd()
    h5_path = Path(INPUT_H5AD) if Path(INPUT_H5AD).is_absolute() else (root / INPUT_H5AD)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reading {h5_path}")
    adata = ad.read_h5ad(h5_path)

    req = [REPLICATE_COL, GROUP_COL, "age", "sex", "apoE", CELLTYPE_COL]
    miss = [c for c in req if c not in adata.obs.columns]
    if miss:
        raise ValueError(f"Missing obs columns: {miss}")

    sample_meta = build_sample_meta(adata.obs)

    subctypes = sorted(pd.unique(adata.obs[CELLTYPE_COL].astype(str)))
    all_summaries: list[pd.DataFrame] = []
    all_results_keyed: dict[str, dict[str, pd.DataFrame]] = {}

    for ct in subctypes:
        pb_gxs = pseudobulk_one(adata, ct)
        if pb_gxs is None:
            print(f"[SKIP pseudobulk] {ct}")
            rows = [
                {
                    "subcelltype": ct,
                    "design_id": did,
                    "comparison": cmp,
                    "status": "skipped_pseudobulk_not_available",
                }
                for cmp in COMPARISONS
                for did in DESIGN_SPECS
            ]
            all_summaries.append(pd.DataFrame(rows))
            continue

        cnt_sxg = pb_gxs.T.astype(int)
        cnt_sxg.index = cnt_sxg.index.astype(str)
        common = [s for s in cnt_sxg.index if str(s) in sample_meta.index.astype(str)]
        if len(common) < MIN_SAMPLES_STAGE:
            print(f"[SKIP ct] {ct}: insufficient sample overlap with meta")
            rows_o = [
                {
                    "subcelltype": ct,
                    "design_id": did,
                    "comparison": cmp,
                    "status": "skipped_insufficient_meta_overlap",
                }
                for cmp in COMPARISONS
                for did in DESIGN_SPECS
            ]
            all_summaries.append(pd.DataFrame(rows_o))
            continue

        cnt_sxg = cnt_sxg.loc[common].copy().sort_index()
        meta_loc = sample_meta.loc[[str(x) for x in cnt_sxg.index]].copy()

        ct_slug = safe_slug(ct)

        if RUN_PCA_SANITY:
            pca_path = OUT_DIR / "diagnostics" / f"{ct_slug}__pseudobulk_pca_log1p.png"
            pseudobulk_pca_plot(cnt_sxg, meta_loc, pca_path, title_suffix=f" — {ct}")

        for design_id, spec in DESIGN_SPECS.items():
            out = run_pydeseq2_contrasts(
                cnt_sxg.copy(),
                meta_loc,
                design_id,
                spec["formula"],
                list(spec["covars"]),
            )

            diag_dir = OUT_DIR / "diagnostics" / f"{ct_slug}__{design_id}"

            if out is None:
                continue

            sum_part = out["summary"].copy()
            sum_part["subcelltype"] = ct
            all_summaries.append(sum_part)

            if len(out["results"]) == 0:
                continue

            diag_dir.mkdir(parents=True, exist_ok=True)
            for cmp_name, rdf in out["results"].items():
                (diag_dir / f"contrast__{cmp_name}").mkdir(parents=True, exist_ok=True)
                rdf.assign(subcelltype=ct).to_csv(
                    diag_dir / f"contrast__{cmp_name}" / "results_pydeseq2_wald.csv", index=False
                )

            for cmp_name, rdf in out["results"].items():
                outp2 = OUT_DIR / f"{ct_slug}__{design_id}__{cmp_name}.csv"
                rdf.assign(subcelltype=ct, design_id=design_id).to_csv(outp2, index=False)

            all_results_keyed[f"{ct}||{design_id}"] = out["results"]

            if RUN_DECOUPLER and design_id == "group_only":
                sub_ct = adata[adata.obs[CELLTYPE_COL].astype(str) == str(ct)].copy()
                run_decoupler_signor_ulm(sub_ct, pb_gxs, diag_dir, ct_slug)

    if all_summaries:
        summary_tab = pd.concat(all_summaries, ignore_index=True)
    else:
        summary_tab = pd.DataFrame()

    summary_tab.to_csv(OUT_DIR / "DEG_counts_by_design_celltype_contrast.csv", index=False)

    with open(OUT_DIR / "results_all_designs.pkl", "wb") as f:
        pickle.dump(all_results_keyed, f)

    with open(OUT_DIR / "python_session.txt", "w") as f:
        f.write(sys.version + "\n\n")
        for pkg in ("pydeseq2", "decoupler", "anndata", "scanpy", "pandas", "numpy", "scipy"):
            try:
                import importlib.metadata as md

                f.write(f"{pkg}=={md.version(pkg)}\n")
            except Exception:
                f.write(f"{pkg}=unknown\n")

    print(f"Done. Outputs: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
