"""Benchmarking module for DTU detection sensitivity.

Generates synthetic single-cell transcript count matrices with known
ground truth DTU signals at varying effect sizes, then evaluates the
sensitivity and specificity of the DTU detection pipeline.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
from tqdm import tqdm

from ._logging import get_named_logger, wf_parser


def argparser():
    """Create argument parser."""
    parser = wf_parser("benchmark_dtu")

    parser.add_argument(
        "--output_dir", type=Path, default="benchmark_results",
        help="Output directory for benchmark results.")
    parser.add_argument(
        "--n_cells", type=int, default=200,
        help="Number of cells per cluster.")
    parser.add_argument(
        "--n_clusters", type=int, default=2,
        help="Number of clusters.")
    parser.add_argument(
        "--n_genes", type=int, default=100,
        help="Number of multi-isoform genes.")
    parser.add_argument(
        "--n_isoforms", type=int, default=3,
        help="Number of isoforms per gene.")
    parser.add_argument(
        "--n_dtu_genes", type=int, default=20,
        help="Number of genes with planted DTU signal.")
    parser.add_argument(
        "--effect_sizes", type=float, nargs='+',
        default=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
        help="Effect sizes to test (proportion shift).")
    parser.add_argument(
        "--n_replicates", type=int, default=5,
        help="Number of replicates per effect size.")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.")

    return parser


def generate_synthetic_data(
        n_cells_per_cluster, n_clusters, n_genes, n_isoforms,
        n_dtu_genes, effect_size, rng):
    """Generate synthetic transcript count matrix with planted DTU.

    :param n_cells_per_cluster: cells per cluster.
    :param n_clusters: number of clusters.
    :param n_genes: total multi-isoform genes.
    :param n_isoforms: isoforms per gene.
    :param n_dtu_genes: genes with DTU signal.
    :param effect_size: proportion shift for DTU genes (0-1).
    :param rng: numpy random generator.
    :returns: tuple of (count_matrix, barcodes, transcript_names,
        gene_map, cluster_labels, dtu_gene_set).
    """
    n_cells = n_cells_per_cluster * n_clusters
    n_transcripts = n_genes * n_isoforms

    # Build transcript/gene names
    transcript_names = []
    gene_map = {}
    for g in range(n_genes):
        gene_id = f"GENE_{g:04d}"
        for t in range(n_isoforms):
            tx_id = f"TX_{g:04d}.{t}"
            transcript_names.append(tx_id)
            gene_map[tx_id] = gene_id

    barcodes = [f"CELL_{i:05d}-1" for i in range(n_cells)]

    # Assign clusters
    cluster_labels = []
    for c in range(n_clusters):
        cluster_labels.extend([str(c)] * n_cells_per_cluster)

    # Select DTU genes
    dtu_genes = set(range(n_dtu_genes))

    # Generate counts
    # Base: uniform isoform proportions, Poisson counts
    base_rate = 5.0
    data = np.zeros((n_transcripts, n_cells), dtype=np.float32)

    for g in range(n_genes):
        for t in range(n_isoforms):
            tx_idx = g * n_isoforms + t

            if g in dtu_genes:
                # DTU gene: shift proportions between clusters
                for c in range(n_clusters):
                    start = c * n_cells_per_cluster
                    end = start + n_cells_per_cluster

                    if c == 0:
                        # Cluster 0: isoform 0 dominant
                        if t == 0:
                            rate = base_rate * (1 + effect_size)
                        else:
                            rate = base_rate * (1 - effect_size /
                                                (n_isoforms - 1))
                    else:
                        # Other clusters: isoform 1 dominant
                        if t == 1:
                            rate = base_rate * (1 + effect_size)
                        else:
                            rate = base_rate * (1 - effect_size /
                                                (n_isoforms - 1))

                    rate = max(rate, 0.1)
                    data[tx_idx, start:end] = rng.poisson(
                        rate, n_cells_per_cluster)
            else:
                # Non-DTU gene: uniform across clusters
                data[tx_idx, :] = rng.poisson(base_rate, n_cells)

    dtu_gene_set = {f"GENE_{g:04d}" for g in dtu_genes}

    return (data, barcodes, transcript_names, gene_map,
            cluster_labels, dtu_gene_set)


def evaluate_dtu_results(dtu_df, true_dtu_genes, fdr_threshold=0.05):
    """Evaluate DTU detection against ground truth.

    :param dtu_df: DataFrame with columns gene, pvalue_adj.
    :param true_dtu_genes: set of gene IDs with true DTU.
    :param fdr_threshold: significance threshold.
    :returns: dict with TP, FP, FN, TN, sensitivity, specificity, etc.
    """
    if len(dtu_df) == 0:
        return {
            'tp': 0, 'fp': 0, 'fn': len(true_dtu_genes), 'tn': 0,
            'sensitivity': 0.0, 'specificity': 1.0,
            'precision': 0.0, 'f1': 0.0,
        }

    sig_genes = set(
        dtu_df[dtu_df['pvalue_adj'] <= fdr_threshold]['gene'].unique())
    all_tested = set(dtu_df['gene'].unique())

    tp = len(sig_genes & true_dtu_genes)
    fp = len(sig_genes - true_dtu_genes)
    fn = len(true_dtu_genes - sig_genes)
    tn = len((all_tested - sig_genes) - true_dtu_genes)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * sensitivity /
          (precision + sensitivity)
          if (precision + sensitivity) > 0 else 0.0)

    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'sensitivity': round(sensitivity, 4),
        'specificity': round(specificity, 4),
        'precision': round(precision, 4),
        'f1': round(f1, 4),
    }


def main(args):
    """Run DTU benchmark."""
    from .differential_transcript_usage import (
        build_gene_groups, chi_squared_dtu_test,
        correct_pvalues, detect_isoform_switching)

    logger = get_named_logger("Benchmark")
    logger.info("Starting DTU benchmark.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_results = []

    for effect_size in tqdm(args.effect_sizes, desc="Benchmarking", disable=None):
        for rep in range(args.n_replicates):
            logger.info(
                f"Effect size {effect_size}, replicate {rep + 1}/"
                f"{args.n_replicates}")

            data, barcodes, tx_names, gene_map, clusters, true_dtu = \
                generate_synthetic_data(
                    args.n_cells, args.n_clusters, args.n_genes,
                    args.n_isoforms, args.n_dtu_genes,
                    effect_size, rng)

            # Build gene groups
            gene_groups = build_gene_groups(
                tx_names, gene_map, min_isoforms=2)

            # Run DTU (one-vs-rest for cluster 0)
            cluster_arr = np.array(clusters)
            cl0_mask = cluster_arr == '0'

            dtu_rows = []
            for gene_id, tx_indices in gene_groups.items():
                tx_idx = np.array(tx_indices)
                agg_cl0 = data[tx_idx][:, cl0_mask].sum(axis=1)
                agg_rest = data[tx_idx][:, ~cl0_mask].sum(axis=1)

                chi2, pval, effect = chi_squared_dtu_test(
                    agg_cl0, agg_rest)
                dtu_rows.append({
                    'gene': gene_id,
                    'test_statistic': chi2,
                    'pvalue': pval,
                    'effect_size': effect,
                })

            dtu_df = pd.DataFrame(dtu_rows)
            if len(dtu_df) > 0:
                dtu_df['pvalue_adj'] = correct_pvalues(
                    dtu_df['pvalue'].values)

            metrics = evaluate_dtu_results(dtu_df, true_dtu)
            metrics['effect_size'] = effect_size
            metrics['replicate'] = rep
            all_results.append(metrics)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        args.output_dir / "benchmark_results.tsv",
        sep='\t', index=False)

    # Summary: mean metrics per effect size
    summary = results_df.groupby('effect_size').agg({
        'sensitivity': 'mean',
        'specificity': 'mean',
        'precision': 'mean',
        'f1': 'mean',
        'tp': 'mean',
        'fp': 'mean',
    }).reset_index().to_dict('records')

    with open(args.output_dir / "benchmark_summary.json", 'w') as fh:
        json.dump({
            'n_cells_per_cluster': args.n_cells,
            'n_genes': args.n_genes,
            'n_dtu_genes': args.n_dtu_genes,
            'n_replicates': args.n_replicates,
            'results_per_effect_size': summary,
        }, fh, indent=2)

    logger.info("Benchmark complete.")
    for row in summary:
        logger.info(
            f"  effect={row['effect_size']:.1f}: "
            f"sensitivity={row['sensitivity']:.3f}, "
            f"specificity={row['specificity']:.3f}, "
            f"F1={row['f1']:.3f}")
