"""Publication-quality visualization module for IRIS results.

Generates figures from IRIS output files including UMAP embeddings,
volcano plots, heatmaps, bar plots, trajectory streams, Manhattan
plots, and cluster comparison matrices. Supports multiple output
formats and style presets for publication or presentation use.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from ._logging import get_named_logger, wf_parser

logger = get_named_logger("plot")


def argparser():
    """Create argument parser."""
    parser = wf_parser("plot")

    parser.add_argument(
        "--out_dir", type=Path, required=True,
        help="IRIS output directory (from `iris run`).")
    parser.add_argument(
        "--output_dir", type=Path, default="iris_figures",
        help="Directory where figures will be saved.")
    parser.add_argument(
        "--format", type=str, default="png",
        choices=["png", "pdf", "svg"],
        help="Output figure format.")
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Figure resolution in dots per inch.")
    parser.add_argument(
        "--style", type=str, default="publication",
        choices=["publication", "presentation"],
        help="Visual style preset.")

    return parser


# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

def set_style(style):
    """Configure matplotlib rcParams for the chosen style.

    Parameters
    ----------
    style : str
        Either 'publication' (Nature/Cell style – small fonts, tight
        layout) or 'presentation' (larger fonts, wider spacing).
    """
    plt.rcdefaults()

    if style == "publication":
        plt.rcParams.update({
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "figure.figsize": (3.5, 3.0),
            "figure.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "lines.linewidth": 0.8,
            "lines.markersize": 2,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        })
    elif style == "presentation":
        plt.rcParams.update({
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.figsize": (8, 6),
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        })
    else:
        logger.warning(f"Unknown style '{style}', using defaults.")


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def _scatter_umap(ax, x, y, c, cmap=None, title="", categorical=False):
    """Helper to draw a UMAP scatter on the given axes."""
    if categorical:
        categories = sorted(c.unique())
        palette = plt.cm.tab20(np.linspace(0, 1, max(len(categories), 1)))
        for idx, cat in enumerate(categories):
            mask = c == cat
            ax.scatter(
                x[mask], y[mask], s=1, alpha=0.6,
                color=palette[idx % len(palette)], label=str(cat),
                rasterized=True)
        ax.legend(
            markerscale=4, frameon=False, fontsize="x-small",
            loc="center left", bbox_to_anchor=(1.0, 0.5))
    else:
        sc = ax.scatter(
            x, y, c=c, cmap=cmap or "viridis", s=1, alpha=0.6,
            rasterized=True)
        plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_joint_umap(out_dir, save_dir, fmt, dpi):
    """Plot UMAP embeddings colored by cluster, cell type, and diversity.

    Parameters
    ----------
    out_dir : Path
        IRIS output directory containing result files.
    save_dir : Path
        Directory where figures are saved.
    fmt : str
        Image format (png, pdf, svg).
    dpi : int
        Figure resolution.
    """
    umap_path = out_dir / "joint.umap.tsv"
    if not umap_path.exists():
        logger.warning("joint.umap.tsv not found, skipping UMAP plots.")
        return

    umap_df = pd.read_csv(umap_path, sep="\t")
    x = umap_df.iloc[:, 1].values
    y = umap_df.iloc[:, 2].values

    # --- Cluster UMAP ---
    cluster_col = None
    for col in ["cluster", "joint_cluster", "gene_cluster"]:
        if col in umap_df.columns:
            cluster_col = col
            break
    if cluster_col is None and umap_df.shape[1] >= 4:
        cluster_col = umap_df.columns[3]

    if cluster_col is not None:
        fig, ax = plt.subplots()
        _scatter_umap(
            ax, x, y, umap_df[cluster_col],
            title="UMAP — Clusters", categorical=True)
        fig.savefig(save_dir / f"umap_clusters.{fmt}", dpi=dpi)
        plt.close(fig)
        logger.info("Saved umap_clusters.%s", fmt)

    # --- Cell type UMAP ---
    ct_path = out_dir / "cell_type_annotations.tsv"
    if ct_path.exists():
        ct_df = pd.read_csv(ct_path, sep="\t")
        ct_col = "cell_type" if "cell_type" in ct_df.columns else ct_df.columns[-1]
        # Align to UMAP barcodes
        barcode_col = umap_df.columns[0]
        merged = umap_df.merge(ct_df, left_on=barcode_col, right_on=ct_df.columns[0], how="left")
        fig, ax = plt.subplots()
        _scatter_umap(
            ax, x, y, merged[ct_col].fillna("Unknown"),
            title="UMAP — Cell Types", categorical=True)
        fig.savefig(save_dir / f"umap_cell_types.{fmt}", dpi=dpi)
        plt.close(fig)
        logger.info("Saved umap_cell_types.%s", fmt)

    # --- Diversity UMAP ---
    div_path = out_dir / "isoform_diversity.tsv"
    if div_path.exists():
        div_df = pd.read_csv(div_path, sep="\t")
        div_col = "diversity" if "diversity" in div_df.columns else div_df.columns[-1]
        barcode_col = umap_df.columns[0]
        merged = umap_df.merge(div_df, left_on=barcode_col, right_on=div_df.columns[0], how="left")
        fig, ax = plt.subplots()
        _scatter_umap(
            ax, x, y, merged[div_col].fillna(0).values,
            cmap="magma", title="UMAP — Isoform Diversity")
        fig.savefig(save_dir / f"umap_diversity.{fmt}", dpi=dpi)
        plt.close(fig)
        logger.info("Saved umap_diversity.%s", fmt)


def plot_dtu_volcano(out_dir, save_dir, fmt, dpi):
    """Volcano plot of DTU results.

    Parameters
    ----------
    out_dir : Path
        IRIS output directory.
    save_dir : Path
        Directory where figures are saved.
    fmt : str
        Image format.
    dpi : int
        Figure resolution.
    """
    dtu_path = out_dir / "dtu_results.tsv"
    if not dtu_path.exists():
        logger.warning("dtu_results.tsv not found, skipping volcano plot.")
        return

    df = pd.read_csv(dtu_path, sep="\t")

    # Determine column names
    es_col = "effect_size"
    pv_col = "pvalue_adj"
    gene_col = "gene_id"
    for c in df.columns:
        if "effect" in c.lower():
            es_col = c
        if "padj" in c.lower() or "pvalue_adj" in c.lower() or "fdr" in c.lower():
            pv_col = c
        if "gene" in c.lower() and "id" in c.lower():
            gene_col = c
        if "gene_name" in c.lower():
            gene_col = c

    df = df.dropna(subset=[es_col, pv_col])
    neg_log_p = -np.log10(df[pv_col].clip(lower=1e-300))
    significant = df[pv_col] < 0.05

    fig, ax = plt.subplots()
    ax.scatter(
        df.loc[~significant, es_col], neg_log_p[~significant],
        s=3, alpha=0.4, color="grey", rasterized=True, label="NS")
    ax.scatter(
        df.loc[significant, es_col], neg_log_p[significant],
        s=3, alpha=0.6, color="red", rasterized=True, label="FDR < 0.05")

    # Label top 10 genes by significance
    top = df.nsmallest(10, pv_col)
    for _, row in top.iterrows():
        label = str(row.get(gene_col, ""))
        if label:
            ax.annotate(
                label,
                (row[es_col], -np.log10(max(row[pv_col], 1e-300))),
                fontsize=5, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.3))

    ax.set_xlabel("Effect size")
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.set_title("DTU Volcano Plot")
    ax.legend(markerscale=3, frameon=False)
    ax.axhline(-np.log10(0.05), ls="--", lw=0.5, color="grey")
    fig.savefig(save_dir / f"dtu_volcano.{fmt}", dpi=dpi)
    plt.close(fig)
    logger.info("Saved dtu_volcano.%s", fmt)


def plot_isoform_usage_heatmap(out_dir, save_dir, fmt, dpi):
    """Heatmap of isoform proportions per cluster for top switching genes.

    Parameters
    ----------
    out_dir : Path
        IRIS output directory.
    save_dir : Path
        Directory where figures are saved.
    fmt : str
        Image format.
    dpi : int
        Figure resolution.
    """
    switch_path = out_dir / "isoform_switching.tsv"
    if not switch_path.exists():
        logger.warning("isoform_switching.tsv not found, skipping heatmap.")
        return

    sw_df = pd.read_csv(switch_path, sep="\t")

    # Identify gene and cluster columns
    gene_col = next((c for c in sw_df.columns if "gene" in c.lower()), sw_df.columns[0])
    cluster_col = next((c for c in sw_df.columns if "cluster" in c.lower()), None)
    tx_col = next((c for c in sw_df.columns if "transcript" in c.lower()), None)
    prop_col = next(
        (c for c in sw_df.columns if "proportion" in c.lower() or "usage" in c.lower()),
        None)

    if cluster_col is None or prop_col is None:
        logger.warning(
            "isoform_switching.tsv missing expected columns, skipping heatmap.")
        return

    # Select top 20 switching genes by occurrence count
    top_genes = sw_df[gene_col].value_counts().head(20).index.tolist()
    sub = sw_df[sw_df[gene_col].isin(top_genes)].copy()

    # Build label combining gene and transcript
    if tx_col:
        sub["_label"] = sub[gene_col].astype(str) + " | " + sub[tx_col].astype(str)
    else:
        sub["_label"] = sub[gene_col].astype(str)

    pivot = sub.pivot_table(
        index="_label", columns=cluster_col, values=prop_col,
        aggfunc="mean", fill_value=0)

    fig, ax = plt.subplots(figsize=(max(4, pivot.shape[1] * 0.6),
                                     max(4, pivot.shape[0] * 0.3)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=5)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Gene | Transcript")
    ax.set_title("Isoform Usage — Top Switching Genes")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Proportion")
    fig.savefig(save_dir / f"isoform_heatmap.{fmt}", dpi=dpi)
    plt.close(fig)
    logger.info("Saved isoform_heatmap.%s", fmt)


def plot_novel_isoform_barplot(out_dir, save_dir, fmt, dpi):
    """Bar plot of novel isoform class code distribution.

    Parameters
    ----------
    out_dir : Path
        IRIS output directory.
    save_dir : Path
        Directory where figures are saved.
    fmt : str
        Image format.
    dpi : int
        Figure resolution.
    """
    novel_path = out_dir / "novel_isoform_summary.json"
    if not novel_path.exists():
        logger.warning(
            "novel_isoform_summary.json not found, skipping barplot.")
        return

    with open(novel_path) as fh:
        summary = json.load(fh)

    # Extract class code distribution
    class_codes = summary.get(
        "class_code_distribution",
        summary.get("class_code_counts", None))
    if not isinstance(class_codes, dict) or not class_codes:
        logger.warning("No class_code_distribution in summary, skipping.")
        return

    codes = sorted(class_codes.keys())
    counts = [class_codes[c] for c in codes]

    color_map = {
        "j": "#1f77b4", "o": "#ff7f0e", "u": "#2ca02c",
        "x": "#d62728", "i": "#9467bd", "y": "#8c564b",
        "p": "#e377c2", "e": "#7f7f7f", "s": "#bcbd22",
        "k": "#17becf",
    }
    colors = [color_map.get(c, "#aaaaaa") for c in codes]

    fig, ax = plt.subplots()
    ax.bar(codes, counts, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Class Code")
    ax.set_ylabel("Count")
    ax.set_title("Novel Isoform Classification")
    for i, (code, cnt) in enumerate(zip(codes, counts)):
        ax.text(i, cnt, str(cnt), ha="center", va="bottom", fontsize=5)
    fig.savefig(save_dir / f"novel_class_codes.{fmt}", dpi=dpi)
    plt.close(fig)
    logger.info("Saved novel_class_codes.%s", fmt)


def plot_trajectory_stream(out_dir, save_dir, fmt, dpi):
    """UMAP colored by pseudotime values.

    Parameters
    ----------
    out_dir : Path
        IRIS output directory.
    save_dir : Path
        Directory where figures are saved.
    fmt : str
        Image format.
    dpi : int
        Figure resolution.
    """
    pt_path = out_dir / "pseudotime.tsv"
    umap_path = out_dir / "joint.umap.tsv"
    if not pt_path.exists():
        logger.warning("pseudotime.tsv not found, skipping trajectory plot.")
        return
    if not umap_path.exists():
        logger.warning("joint.umap.tsv not found, skipping trajectory plot.")
        return

    umap_df = pd.read_csv(umap_path, sep="\t")
    pt_df = pd.read_csv(pt_path, sep="\t")

    barcode_col = umap_df.columns[0]
    pt_col = next(
        (c for c in pt_df.columns if "pseudotime" in c.lower() or "time" in c.lower()),
        pt_df.columns[-1])

    merged = umap_df.merge(
        pt_df, left_on=barcode_col, right_on=pt_df.columns[0], how="left")

    x = merged.iloc[:, 1].values
    y = merged.iloc[:, 2].values
    pt_vals = merged[pt_col].fillna(0).values

    fig, ax = plt.subplots()
    order = np.argsort(pt_vals)
    sc = ax.scatter(
        x[order], y[order], c=pt_vals[order],
        cmap="viridis", s=1, alpha=0.7, rasterized=True)
    plt.colorbar(sc, ax=ax, shrink=0.6, label="Pseudotime")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Trajectory — Pseudotime")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(save_dir / f"trajectory_pseudotime.{fmt}", dpi=dpi)
    plt.close(fig)
    logger.info("Saved trajectory_pseudotime.%s", fmt)


def plot_ase_manhattan(out_dir, save_dir, fmt, dpi):
    """Manhattan-style plot for allele-specific expression results.

    Parameters
    ----------
    out_dir : Path
        IRIS output directory.
    save_dir : Path
        Directory where figures are saved.
    fmt : str
        Image format.
    dpi : int
        Figure resolution.
    """
    ase_path = out_dir / "ase_results.tsv"
    if not ase_path.exists():
        logger.warning("ase_results.tsv not found, skipping Manhattan plot.")
        return

    df = pd.read_csv(ase_path, sep="\t")

    # Identify columns
    chr_col = next(
        (c for c in df.columns if c.lower() in ("chr", "chrom", "chromosome")),
        None)
    pos_col = next(
        (c for c in df.columns if c.lower() in ("pos", "position", "start")),
        None)
    pv_col = next(
        (c for c in df.columns
         if "padj" in c.lower() or "pvalue_adj" in c.lower() or "fdr" in c.lower()),
        None)

    if chr_col is None or pos_col is None or pv_col is None:
        logger.warning(
            "ase_results.tsv missing chr/pos/pvalue_adj columns, "
            "skipping Manhattan plot.")
        return

    df = df.dropna(subset=[chr_col, pos_col, pv_col]).copy()
    df[pos_col] = df[pos_col].astype(int)

    # Sort by chromosome and position
    chrom_order = []
    for c in df[chr_col].unique():
        try:
            chrom_order.append((int(str(c).replace("chr", "")), c))
        except ValueError:
            chrom_order.append((999, c))
    chrom_order.sort()
    chrom_list = [c for _, c in chrom_order]

    df["_chrom_idx"] = df[chr_col].map(
        {c: i for i, c in enumerate(chrom_list)})
    df = df.sort_values(["_chrom_idx", pos_col])

    # Compute cumulative position for x-axis
    cumulative_offset = {}
    running = 0
    for chrom in chrom_list:
        cumulative_offset[chrom] = running
        sub = df[df[chr_col] == chrom]
        if len(sub):
            running += sub[pos_col].max() + 1_000_000

    df["_cum_pos"] = df.apply(
        lambda r: r[pos_col] + cumulative_offset[r[chr_col]], axis=1)
    neg_log_p = -np.log10(df[pv_col].clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=(8, 3))
    palette = plt.cm.tab20(np.linspace(0, 1, len(chrom_list)))
    for idx, chrom in enumerate(chrom_list):
        mask = df[chr_col] == chrom
        ax.scatter(
            df.loc[mask, "_cum_pos"], neg_log_p[mask],
            s=2, alpha=0.5, color=palette[idx % len(palette)],
            rasterized=True)

    ax.set_xlabel("Genomic position")
    ax.set_ylabel("-log10(adjusted p-value)")
    ax.set_title("ASE Manhattan Plot")
    ax.axhline(-np.log10(0.05), ls="--", lw=0.5, color="red", label="FDR=0.05")

    # Chromosome labels at midpoints
    tick_pos = []
    tick_labels = []
    for chrom in chrom_list:
        sub = df[df[chr_col] == chrom]
        if len(sub):
            mid = sub["_cum_pos"].median()
            tick_pos.append(mid)
            tick_labels.append(str(chrom))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=5)
    fig.savefig(save_dir / f"ase_manhattan.{fmt}", dpi=dpi)
    plt.close(fig)
    logger.info("Saved ase_manhattan.%s", fmt)


def plot_cluster_comparison(out_dir, save_dir, fmt, dpi):
    """Heatmap of gene-cluster vs isoform-cluster contingency table.

    Parameters
    ----------
    out_dir : Path
        IRIS output directory.
    save_dir : Path
        Directory where figures are saved.
    fmt : str
        Image format.
    dpi : int
        Figure resolution.
    """
    comp_path = out_dir / "cluster_comparison.json"
    if not comp_path.exists():
        logger.warning(
            "cluster_comparison.json not found, skipping comparison heatmap.")
        return

    with open(comp_path) as fh:
        comparison = json.load(fh)

    ct = comparison.get("contingency_table", None)
    if ct is None:
        logger.warning(
            "No contingency_table in cluster_comparison.json, skipping.")
        return

    gene_labels = ct.get("gene_clusters", [])
    iso_labels = ct.get("isoform_clusters", [])
    counts = ct.get("counts", [])
    if not counts:
        logger.warning("Empty contingency table, skipping.")
        return

    matrix = pd.DataFrame(
        counts, index=gene_labels, columns=iso_labels).fillna(0)

    fig, ax = plt.subplots(
        figsize=(max(4, matrix.shape[1] * 0.8),
                 max(3, matrix.shape[0] * 0.6)))
    im = ax.imshow(matrix.values.astype(float), aspect="auto", cmap="Blues")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Isoform Cluster")
    ax.set_ylabel("Gene Cluster")
    ax.set_title("Cluster Contingency Table")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Cell count")

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = int(matrix.iloc[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=5)

    fig.savefig(save_dir / f"cluster_contingency.{fmt}", dpi=dpi)
    plt.close(fig)
    logger.info("Saved cluster_contingency.%s", fmt)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(args):
    """Generate all available IRIS figures.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments with out_dir, output_dir, format,
        dpi, and style fields.
    """
    logger.info("IRIS Plot — generating figures")

    out_dir = Path(args.out_dir)
    save_dir = Path(args.output_dir)
    fmt = args.format
    dpi = args.dpi

    if not out_dir.exists():
        logger.error("IRIS output directory does not exist: %s", out_dir)
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    set_style(args.style)

    plot_functions = [
        ("Joint UMAP", plot_joint_umap),
        ("DTU Volcano", plot_dtu_volcano),
        ("Isoform Heatmap", plot_isoform_usage_heatmap),
        ("Novel Isoform Barplot", plot_novel_isoform_barplot),
        ("Trajectory Stream", plot_trajectory_stream),
        ("ASE Manhattan", plot_ase_manhattan),
        ("Cluster Comparison", plot_cluster_comparison),
    ]

    generated = []
    for name, func in plot_functions:
        try:
            func(out_dir, save_dir, fmt, dpi)
            generated.append(name)
        except Exception as exc:
            logger.warning("Failed to generate %s: %s", name, exc)

    logger.info(
        "Finished. Generated %d/%d figure sets: %s",
        len(generated), len(plot_functions), ", ".join(generated))
    logger.info("Figures saved to: %s", save_dir.resolve())
