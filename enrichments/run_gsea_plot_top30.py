"""Re-plot GSEApy prerank NES barplots as top-30 (15 up + 15 down).

Selection per panel (cell type x contrast x library), from the full *_gsea.csv:
  up   -> NES > 0, sort NES descending, keep top 15 (most strongly up)
  down -> NES < 0, sort NES ascending,  keep top 15 (most strongly down)
  combine -> at most 20 bars, most-positive at top ... most-negative at bottom.

No significance filter is applied to the *selection* (extremes by NES).
Colour encodes significance by raw pval:
  red  = pval < 0.05
  blue = pval >= 0.05 (or missing)

Figures go to GSEA_prerank_<run>/plots_top30/ with the same basenames as the
originals so the deck builder can consume them unchanged.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parent
RUNS = ["wald", "log2fc"]

N_PER_SIDE = 15
PVAL_CUT = 0.05

SIG_COLOR = "#e41a1c"    # pval < 0.05
NONSIG_COLOR = "#377eb8"  # pval >= 0.05


def select_top30(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["NES"] = pd.to_numeric(df["NES"], errors="coerce")
    df = df.dropna(subset=["NES"])
    up = df[df["NES"] > 0].sort_values("NES", ascending=False).head(N_PER_SIDE)
    down = df[df["NES"] < 0].sort_values("NES", ascending=True).head(N_PER_SIDE)
    # Order for display: most positive first ... most negative last.
    return pd.concat([up.sort_values("NES", ascending=False),
                      down.sort_values("NES", ascending=False)])


def barplot_top30(df: pd.DataFrame, out_png: Path, title: str) -> None:
    plot_df = select_top30(df)
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No pathways", ha="center", va="center", fontsize=13)
        ax.set_title(title)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    nes = plot_df["NES"].values
    pval = pd.to_numeric(plot_df.get("pval"), errors="coerce")
    colors = [SIG_COLOR if (p is not None and pd.notna(p) and p < PVAL_CUT) else NONSIG_COLOR
              for p in pval]
    n_up = int((nes > 0).sum())
    n_down = int((nes < 0).sum())
    n_sig = int((pd.to_numeric(plot_df.get("pval"), errors="coerce") < PVAL_CUT).sum())
    y_pos = range(len(plot_df))

    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.30)))
    ax.barh(list(y_pos), nes, color=colors)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(plot_df["pathway"].tolist(), fontsize=7)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("NES  (red = pval < %.2g,  blue = pval >= %.2g)" % (PVAL_CUT, PVAL_CUT))
    ax.set_title("%s\n(top %d up / %d down;  %d up, %d down shown, %d with pval<%.2g)"
                 % (title, N_PER_SIDE, N_PER_SIDE, n_up, n_down, n_sig, PVAL_CUT),
                 fontsize=10)
    ax.invert_yaxis()  # index 0 (most positive) at top
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for run in RUNS:
        tables = BASE / f"GSEA_prerank_{run}" / "tables"
        out_dir = BASE / f"GSEA_prerank_{run}" / "plots_top30"
        out_dir.mkdir(parents=True, exist_ok=True)
        csvs = sorted(tables.glob("*_gsea.csv"))
        n_ok = 0
        for csv_path in csvs:
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:  # noqa: BLE001
                print("skip (read error):", csv_path.name, exc)
                continue
            if "NES" not in df.columns or "pathway" not in df.columns:
                print("skip (missing cols):", csv_path.name)
                continue
            label = csv_path.stem.removesuffix("_gsea")
            out_png = out_dir / f"{label}_NESbar.png"
            barplot_top30(df, out_png, label.replace("__", " | "))
            n_ok += 1
        print(f"[{run}] wrote {n_ok} figures to {out_dir}")


if __name__ == "__main__":
    main()
