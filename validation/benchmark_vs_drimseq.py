#!/usr/bin/env python3
"""Benchmark IRIS DTU detection against DRIMSeq.

Generates synthetic single-cell data with known ground truth DTU,
runs both IRIS and DRIMSeq (if R is available), and compares results.

DRIMSeq is an R/Bioconductor package for differential transcript usage.
To install DRIMSeq in R:

    if (!requireNamespace("BiocManager", quietly = TRUE))
        install.packages("BiocManager")
    BiocManager::install("DRIMSeq")

Usage:
    python validation/benchmark_vs_drimseq.py --output_dir validation/benchmark_results/

The script always runs IRIS and reports its performance metrics.  If an R
installation with DRIMSeq is found, it will also run DRIMSeq on the same
synthetic data and produce a head-to-head comparison.  If R/DRIMSeq is absent
the comparison step is silently skipped.

Output files (all written to --output_dir):
    iris_results_es{effect_size}_rep{rep}.tsv  - per-replicate IRIS DTU table
    drimseq_input_es{effect_size}_rep{rep}.csv - count matrix for DRIMSeq
    run_drimseq_es{effect_size}_rep{rep}.R      - generated R script
    drimseq_results_es{effect_size}_rep{rep}.csv - DRIMSeq output (if run)
    benchmark_full_results.tsv                 - all per-replicate metrics
    benchmark_summary.json                     - mean metrics per effect size
"""
import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BenchmarkVsDRIMSeq")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def argparser():
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("validation/benchmark_results"),
        help="Directory for all output files.  Created if absent. "
             "(default: %(default)s)")
    parser.add_argument(
        "--n_cells", type=int, default=200,
        help="Cells per cluster in synthetic data. (default: %(default)s)")
    parser.add_argument(
        "--n_clusters", type=int, default=2,
        help="Number of clusters. (default: %(default)s)")
    parser.add_argument(
        "--n_genes", type=int, default=100,
        help="Total multi-isoform genes. (default: %(default)s)")
    parser.add_argument(
        "--n_isoforms", type=int, default=3,
        help="Isoforms per gene. (default: %(default)s)")
    parser.add_argument(
        "--n_dtu_genes", type=int, default=20,
        help="Genes with planted DTU signal. (default: %(default)s)")
    parser.add_argument(
        "--effect_sizes", type=float, nargs="+",
        default=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
        help="Proportion-shift effect sizes to evaluate. "
             "(default: %(default)s)")
    parser.add_argument(
        "--n_replicates", type=int, default=5,
        help="Replicates per effect size. (default: %(default)s)")
    parser.add_argument(
        "--fdr_threshold", type=float, default=0.05,
        help="FDR significance threshold. (default: %(default)s)")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed. (default: %(default)s)")
    parser.add_argument(
        "--skip_drimseq", action="store_true",
        help="Skip DRIMSeq even if R is available (IRIS-only run).")
    return parser


# ---------------------------------------------------------------------------
# R / DRIMSeq availability probe
# ---------------------------------------------------------------------------

def _check_r_available():
    """Return True if the 'Rscript' binary is on PATH."""
    return shutil.which("Rscript") is not None


def _check_drimseq_installed():
    """Return True if DRIMSeq can be loaded inside R."""
    try:
        result = subprocess.run(
            ["Rscript", "-e",
             'suppressPackageStartupMessages(library(DRIMSeq)); cat("ok")'],
            capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# Synthetic data generation  (thin wrapper around iris.benchmark_dtu)
# ---------------------------------------------------------------------------

def generate_data(n_cells, n_clusters, n_genes, n_isoforms,
                  n_dtu_genes, effect_size, rng):
    """Wrap iris.benchmark_dtu.generate_synthetic_data.

    :returns: (data, barcodes, tx_names, gene_map, cluster_labels,
               true_dtu_gene_set)
    """
    from iris.benchmark_dtu import generate_synthetic_data
    return generate_synthetic_data(
        n_cells_per_cluster=n_cells,
        n_clusters=n_clusters,
        n_genes=n_genes,
        n_isoforms=n_isoforms,
        n_dtu_genes=n_dtu_genes,
        effect_size=effect_size,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# IRIS DTU runner
# ---------------------------------------------------------------------------

def run_iris_dtu(data, tx_names, gene_map, cluster_labels, fdr_threshold):
    """Run IRIS chi-squared DTU test (cluster 0 vs rest).

    :param data: ndarray (n_transcripts x n_cells), float counts.
    :param tx_names: list of transcript names (length n_transcripts).
    :param gene_map: dict mapping transcript_id -> gene_id.
    :param cluster_labels: list of cluster label strings (length n_cells).
    :param fdr_threshold: FDR significance cut-off.
    :returns: DataFrame with columns gene, pvalue, pvalue_adj, effect_size,
              test_statistic.
    """
    from iris.differential_transcript_usage import (
        build_gene_groups,
        chi_squared_dtu_test,
        correct_pvalues,
    )

    cluster_arr = np.array(cluster_labels)
    cl0_mask = cluster_arr == "0"

    gene_groups = build_gene_groups(tx_names, gene_map, min_isoforms=2)

    rows = []
    for gene_id, tx_indices in gene_groups.items():
        tx_idx = np.array(tx_indices)
        agg_cl0 = data[tx_idx][:, cl0_mask].sum(axis=1)
        agg_rest = data[tx_idx][:, ~cl0_mask].sum(axis=1)

        # chi_squared_dtu_test returns (chi2, pvalue, effect_size,
        # convergence_info) — unpack all four, discard convergence_info.
        chi2, pval, effect, _conv = chi_squared_dtu_test(agg_cl0, agg_rest)
        rows.append({
            "gene": gene_id,
            "test_statistic": chi2,
            "pvalue": pval,
            "effect_size": effect,
        })

    dtu_df = pd.DataFrame(rows)
    if len(dtu_df) > 0:
        dtu_df["pvalue_adj"] = correct_pvalues(dtu_df["pvalue"].values)
    else:
        dtu_df["pvalue_adj"] = pd.Series(dtype=float)

    return dtu_df


# ---------------------------------------------------------------------------
# DRIMSeq data export
# ---------------------------------------------------------------------------

def export_drimseq_input(data, barcodes, tx_names, gene_map,
                         cluster_labels, csv_path):
    """Write a long-format count CSV suitable for DRIMSeq.

    Columns: feature_id, gene_id, <sample_id_per_cluster_aggregate>

    DRIMSeq expects one row per transcript and one numeric column per
    sample.  Here we treat each cell as a sample (a common single-cell
    pseudo-bulk approach).  We aggregate counts per cluster to keep the
    CSV tractable and clearly separate the two conditions.

    The CSV layout is:
        feature_id  gene_id  cluster_0  cluster_1  ... cluster_N

    :param data: ndarray (n_transcripts x n_cells).
    :param barcodes: list of cell barcode strings.
    :param tx_names: list of transcript names.
    :param gene_map: dict transcript_id -> gene_id.
    :param cluster_labels: list of cluster label strings.
    :param csv_path: Path where the CSV should be written.
    """
    cluster_arr = np.array(cluster_labels)
    unique_clusters = sorted(set(cluster_labels))

    rows = []
    for tx_idx, tx_name in enumerate(tx_names):
        gene_id = gene_map.get(tx_name, "UNKNOWN")
        row = {"feature_id": tx_name, "gene_id": gene_id}
        for cl in unique_clusters:
            mask = cluster_arr == cl
            row[f"cluster_{cl}"] = int(data[tx_idx, mask].sum())
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logger.debug(f"DRIMSeq input written to {csv_path} "
                 f"({len(df)} transcripts).")


# ---------------------------------------------------------------------------
# R script generation
# ---------------------------------------------------------------------------

_DRIMSEQ_R_TEMPLATE = """\
#!/usr/bin/env Rscript
# Auto-generated by benchmark_vs_drimseq.py
# DRIMSeq analysis for effect_size={effect_size}, replicate={replicate}
#
# Installation (run once in R):
#   if (!requireNamespace("BiocManager", quietly = TRUE))
#       install.packages("BiocManager")
#   BiocManager::install("DRIMSeq")

suppressPackageStartupMessages({{
    library(DRIMSeq)
    library(BiocParallel)
}}

# ---- Load counts ---------------------------------------------------------
counts <- read.csv("{csv_path}", stringsAsFactors = FALSE)

# Identify sample (cluster) columns
sample_cols <- grep("^cluster_", colnames(counts), value = TRUE)
count_mat   <- as.matrix(counts[, sample_cols])
rownames(count_mat) <- counts$feature_id

# ---- Build DRIMSeq dmDSdata object --------------------------------------
# Sample metadata: one row per pseudo-bulk column
sample_meta <- data.frame(
    sample_id  = sample_cols,
    condition  = ifelse(sample_cols == "cluster_0", "A", "B"),
    stringsAsFactors = FALSE
)
rownames(sample_meta) <- sample_cols

# Transcript-level metadata
tx_meta <- data.frame(
    feature_id = counts$feature_id,
    gene_id    = counts$gene_id,
    stringsAsFactors = FALSE
)

d <- dmDSdata(counts = count_mat, samples = sample_meta,
              annotation = tx_meta)

# ---- Filter -------------------------------------------------------------
d <- dmFilter(d,
              min_samps_feature_expr  = 1,
              min_feature_expr        = 1,
              min_samps_feature_prop  = 1,
              min_feature_prop        = 0.01)

# ---- Precision estimation -----------------------------------------------
set.seed({seed})
d <- dmPrecision(d, BPPARAM = SerialParam())

# ---- Fit models ---------------------------------------------------------
design <- model.matrix(~ condition, data = samples(d))
d <- dmFit(d, design = design, BPPARAM = SerialParam())

# ---- Test ---------------------------------------------------------------
d <- dmTest(d, coef = "conditionB", BPPARAM = SerialParam())

# ---- Export results -----------------------------------------------------
res <- results(d, level = "gene")
write.csv(res, file = "{output_path}", row.names = FALSE, quote = FALSE)

cat("DRIMSeq complete. Significant genes (FDR <= 0.05):",
    sum(res$adj_pvalue <= 0.05, na.rm = TRUE), "\\n")
"""


def write_drimseq_r_script(effect_size, replicate, csv_path, output_path,
                            r_script_path, seed):
    """Write an R script that runs DRIMSeq on the exported CSV.

    :param effect_size: numeric effect size (for comment in script).
    :param replicate: replicate index (for comment).
    :param csv_path: absolute path to the DRIMSeq input CSV.
    :param output_path: absolute path for DRIMSeq results CSV.
    :param r_script_path: Path where the .R file should be saved.
    :param seed: integer random seed for DRIMSeq precision estimation.
    """
    script = _DRIMSEQ_R_TEMPLATE.format(
        effect_size=effect_size,
        replicate=replicate,
        csv_path=str(csv_path),
        output_path=str(output_path),
        seed=seed,
    )
    r_script_path.write_text(script)
    logger.debug(f"R script written to {r_script_path}.")


# ---------------------------------------------------------------------------
# DRIMSeq runner
# ---------------------------------------------------------------------------

def run_drimseq(r_script_path, timeout=300):
    """Execute the generated R script via Rscript.

    :param r_script_path: Path to the .R script.
    :param timeout: seconds before the subprocess is killed.
    :returns: True on success, False on failure.
    """
    try:
        result = subprocess.run(
            ["Rscript", "--vanilla", str(r_script_path)],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            logger.warning(
                f"DRIMSeq R script exited with code {result.returncode}.\n"
                f"STDERR: {result.stderr[-2000:]}")
            return False
        logger.debug(f"DRIMSeq stdout: {result.stdout[-1000:]}")
        return True
    except subprocess.TimeoutExpired:
        logger.warning(f"DRIMSeq R script timed out after {timeout}s.")
        return False
    except FileNotFoundError:
        logger.warning("Rscript not found on PATH.")
        return False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(result_df, true_dtu_genes, fdr_threshold=0.05,
                    gene_col="gene", padj_col="pvalue_adj"):
    """Compute sensitivity, specificity, precision, and F1.

    :param result_df: DataFrame with at least gene_col and padj_col.
    :param true_dtu_genes: set of gene IDs with planted DTU.
    :param fdr_threshold: significance cut-off.
    :param gene_col: column name for gene IDs.
    :param padj_col: column name for adjusted p-values.
    :returns: dict with TP, FP, FN, TN, sensitivity, specificity,
              precision, f1.
    """
    from iris.benchmark_dtu import evaluate_dtu_results

    # Normalise column names before handing off to the shared evaluator
    eval_df = result_df.rename(
        columns={gene_col: "gene", padj_col: "pvalue_adj"})
    return evaluate_dtu_results(eval_df, true_dtu_genes,
                                fdr_threshold=fdr_threshold)


def load_drimseq_results(csv_path):
    """Read DRIMSeq gene-level results CSV.

    :param csv_path: Path to the DRIMSeq output CSV.
    :returns: DataFrame or None if file is absent / unreadable.
    """
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as exc:
        logger.warning(f"Could not read DRIMSeq output {csv_path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _mean_metrics(records):
    """Average numeric metric fields across a list of dicts."""
    if not records:
        return {}
    keys = [k for k in records[0] if isinstance(records[0][k], (int, float))]
    return {k: round(float(np.mean([r[k] for r in records])), 4) for k in keys}


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def main(args):
    """Run the full benchmark."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Probe R / DRIMSeq availability once up-front
    use_drimseq = False
    if not args.skip_drimseq:
        if _check_r_available():
            logger.info("R found on PATH.  Checking for DRIMSeq package...")
            if _check_drimseq_installed():
                logger.info("DRIMSeq is installed.  Head-to-head comparison "
                            "will be run.")
                use_drimseq = True
            else:
                logger.warning(
                    "DRIMSeq not found in R library.  "
                    "IRIS-only mode.\n"
                    "To install DRIMSeq:\n"
                    "  if (!requireNamespace('BiocManager', quietly=TRUE))\n"
                    "      install.packages('BiocManager')\n"
                    "  BiocManager::install('DRIMSeq')")
        else:
            logger.warning(
                "Rscript not found on PATH.  IRIS-only mode.\n"
                "Install R from https://cran.r-project.org/ and then "
                "DRIMSeq via BiocManager to enable the comparison.")
    else:
        logger.info("--skip_drimseq set.  Running IRIS only.")

    # Results containers
    all_iris_metrics = []
    all_drimseq_metrics = []
    full_rows = []  # one row per (effect_size, replicate, method)

    for effect_size in args.effect_sizes:
        iris_reps = []
        drimseq_reps = []

        for rep in range(args.n_replicates):
            tag = f"es{effect_size}_rep{rep}"
            logger.info(
                f"Effect size {effect_size:.1f}  replicate {rep + 1}/"
                f"{args.n_replicates}")

            # ------------------------------------------------------------------
            # 1. Generate synthetic data
            # ------------------------------------------------------------------
            data, barcodes, tx_names, gene_map, cluster_labels, true_dtu = \
                generate_data(
                    n_cells=args.n_cells,
                    n_clusters=args.n_clusters,
                    n_genes=args.n_genes,
                    n_isoforms=args.n_isoforms,
                    n_dtu_genes=args.n_dtu_genes,
                    effect_size=effect_size,
                    rng=rng,
                )

            # ------------------------------------------------------------------
            # 2. Run IRIS DTU
            # ------------------------------------------------------------------
            iris_df = run_iris_dtu(
                data, tx_names, gene_map, cluster_labels,
                fdr_threshold=args.fdr_threshold,
            )

            iris_out = args.output_dir / f"iris_results_{tag}.tsv"
            iris_df.to_csv(iris_out, sep="\t", index=False)

            iris_metrics = compute_metrics(
                iris_df, true_dtu, fdr_threshold=args.fdr_threshold)
            iris_metrics.update(
                effect_size=effect_size, replicate=rep, method="IRIS")
            iris_reps.append(iris_metrics)
            full_rows.append(iris_metrics.copy())

            logger.info(
                f"  IRIS  — sens={iris_metrics['sensitivity']:.3f}  "
                f"spec={iris_metrics['specificity']:.3f}  "
                f"F1={iris_metrics['f1']:.3f}")

            # ------------------------------------------------------------------
            # 3. Export data for DRIMSeq
            # ------------------------------------------------------------------
            csv_path = args.output_dir / f"drimseq_input_{tag}.csv"
            export_drimseq_input(
                data, barcodes, tx_names, gene_map, cluster_labels, csv_path)

            # ------------------------------------------------------------------
            # 4. Generate R script
            # ------------------------------------------------------------------
            drimseq_out = args.output_dir / f"drimseq_results_{tag}.csv"
            r_script_path = args.output_dir / f"run_drimseq_{tag}.R"
            write_drimseq_r_script(
                effect_size=effect_size,
                replicate=rep,
                csv_path=csv_path.resolve(),
                output_path=drimseq_out.resolve(),
                r_script_path=r_script_path,
                seed=args.seed + rep,
            )

            # ------------------------------------------------------------------
            # 5. Run DRIMSeq (if available) and parse results
            # ------------------------------------------------------------------
            if use_drimseq:
                success = run_drimseq(r_script_path)
                dr_df = load_drimseq_results(drimseq_out) if success else None

                if dr_df is not None and len(dr_df) > 0:
                    # DRIMSeq gene-level results use column 'gene_id' and
                    # 'adj_pvalue' (v1.x).  Guard against variation.
                    gene_col = next(
                        (c for c in ("gene_id", "gene", "featureID")
                         if c in dr_df.columns), dr_df.columns[0])
                    padj_col = next(
                        (c for c in ("adj_pvalue", "padj", "FDR",
                                     "adjusted.pvalue")
                         if c in dr_df.columns), None)

                    if padj_col is not None:
                        dr_metrics = compute_metrics(
                            dr_df, true_dtu,
                            fdr_threshold=args.fdr_threshold,
                            gene_col=gene_col, padj_col=padj_col)
                        dr_metrics.update(
                            effect_size=effect_size,
                            replicate=rep,
                            method="DRIMSeq",
                        )
                        drimseq_reps.append(dr_metrics)
                        full_rows.append(dr_metrics.copy())

                        logger.info(
                            f"  DRIMSeq — sens={dr_metrics['sensitivity']:.3f}"
                            f"  spec={dr_metrics['specificity']:.3f}  "
                            f"F1={dr_metrics['f1']:.3f}")
                    else:
                        logger.warning(
                            "Could not identify adjusted p-value column "
                            f"in DRIMSeq output.  Columns: {dr_df.columns.tolist()}")
                else:
                    logger.warning(
                        f"DRIMSeq produced no usable results for {tag}.")

        # Per-effect-size summary entry
        all_iris_metrics.append(
            {"effect_size": effect_size,
             "method": "IRIS",
             **_mean_metrics(iris_reps)})
        if drimseq_reps:
            all_drimseq_metrics.append(
                {"effect_size": effect_size,
                 "method": "DRIMSeq",
                 **_mean_metrics(drimseq_reps)})

    # --------------------------------------------------------------------------
    # 6. Write full per-replicate results table
    # --------------------------------------------------------------------------
    full_df = pd.DataFrame(full_rows)
    full_tsv = args.output_dir / "benchmark_full_results.tsv"
    full_df.to_csv(full_tsv, sep="\t", index=False)
    logger.info(f"Full per-replicate results written to {full_tsv}.")

    # --------------------------------------------------------------------------
    # 7. Write JSON summary
    # --------------------------------------------------------------------------
    summary = {
        "settings": {
            "n_cells_per_cluster": args.n_cells,
            "n_clusters": args.n_clusters,
            "n_genes": args.n_genes,
            "n_isoforms": args.n_isoforms,
            "n_dtu_genes": args.n_dtu_genes,
            "n_replicates": args.n_replicates,
            "fdr_threshold": args.fdr_threshold,
            "effect_sizes": args.effect_sizes,
            "seed": args.seed,
        },
        "drimseq_available": use_drimseq,
        "iris_per_effect_size": all_iris_metrics,
    }
    if all_drimseq_metrics:
        summary["drimseq_per_effect_size"] = all_drimseq_metrics

        # Head-to-head delta table (IRIS minus DRIMSeq, positive = IRIS better)
        iris_by_es = {r["effect_size"]: r for r in all_iris_metrics}
        dr_by_es = {r["effect_size"]: r for r in all_drimseq_metrics}
        deltas = []
        for es in args.effect_sizes:
            if es in iris_by_es and es in dr_by_es:
                ir, dr = iris_by_es[es], dr_by_es[es]
                deltas.append({
                    "effect_size": es,
                    "delta_sensitivity": round(
                        ir["sensitivity"] - dr["sensitivity"], 4),
                    "delta_specificity": round(
                        ir["specificity"] - dr["specificity"], 4),
                    "delta_f1": round(ir["f1"] - dr["f1"], 4),
                })
        summary["iris_minus_drimseq_deltas"] = deltas

    json_out = args.output_dir / "benchmark_summary.json"
    with json_out.open("w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"Summary written to {json_out}.")

    # --------------------------------------------------------------------------
    # 8. Console report
    # --------------------------------------------------------------------------
    _print_report(all_iris_metrics, all_drimseq_metrics, use_drimseq)

    return summary


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def _fmt_row(label, metrics_list):
    """Format a table row showing sensitivity/specificity/F1 per effect size."""
    cells = " ".join(
        f"{r['sensitivity']:.2f}/{r['specificity']:.2f}/{r['f1']:.2f}"
        for r in metrics_list
    )
    return f"  {label:<10} {cells}"


def _print_report(iris_metrics, drimseq_metrics, drimseq_ran):
    """Print a human-readable summary table to stdout."""
    effect_sizes = [r["effect_size"] for r in iris_metrics]

    header_cells = " ".join(f"   ES={es:<4}   " for es in effect_sizes)
    divider = "-" * (12 + 15 * len(effect_sizes))

    print("\n" + divider)
    print("  BENCHMARK SUMMARY   (sens/spec/F1 — mean over replicates)")
    print(f"  {'method':<10} {header_cells}")
    print(divider)
    print(_fmt_row("IRIS", iris_metrics))
    if drimseq_ran and drimseq_metrics:
        print(_fmt_row("DRIMSeq", drimseq_metrics))
    elif not drimseq_ran:
        print("  DRIMSeq    [not run — R/DRIMSeq not available]")
    else:
        print("  DRIMSeq    [no usable results produced]")
    print(divider + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparser()
    args = parser.parse_args()
    main(args)
