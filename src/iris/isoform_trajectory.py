"""Isoform-aware trajectory analysis with pseudotime.

Computes diffusion pseudotime from gene expression, then tracks
isoform proportion dynamics along the trajectory. Detects isoform
switching events where dominant transcript usage changes as cells
progress through a biological process.

Uses Scanpy's diffusion map and DPT implementations. Isoform trends
are assessed by Spearman correlation of per-bin isoform proportions
with pseudotime.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats
from tqdm import tqdm

from ._logging import get_named_logger, wf_parser


def argparser():
    """Create argument parser."""
    parser = wf_parser("isoform_trajectory")

    parser.add_argument(
        "gene_matrix_dir", type=Path,
        help="Path to processed gene MEX matrix directory.")
    parser.add_argument(
        "--transcript_matrix_dir", type=Path, required=True,
        help="Path to processed transcript MEX matrix directory.")
    parser.add_argument(
        "--gene_transcript_map", type=Path, required=True,
        help="TSV mapping transcript_id to gene_id.")
    parser.add_argument(
        "--clusters", type=Path, default=None,
        help="TSV with cluster assignments (optional, for root selection).")
    parser.add_argument(
        "--output_pseudotime", type=Path, default="pseudotime.tsv",
        help="Output TSV with pseudotime per cell.")
    parser.add_argument(
        "--output_isoform_dynamics", type=Path,
        default="isoform_dynamics.tsv",
        help="Output TSV with isoform trend statistics.")
    parser.add_argument(
        "--output_switching_trajectory", type=Path,
        default="trajectory_switching.tsv",
        help="Output TSV with isoform switching events along trajectory.")
    parser.add_argument(
        "--output_summary", type=Path, default="trajectory_summary.json",
        help="Output JSON with trajectory summary statistics.")
    parser.add_argument(
        "--n_dpt_neighbors", type=int, default=15,
        help="Number of neighbors for diffusion map.")
    parser.add_argument(
        "--n_pcs", type=int, default=30,
        help="Number of PCs for dimensionality reduction.")
    parser.add_argument(
        "--min_isoforms", type=int, default=2,
        help="Minimum isoforms per gene to analyze trends.")
    parser.add_argument(
        "--n_bins", type=int, default=10,
        help="Number of pseudotime bins for trend analysis.")
    parser.add_argument(
        "--n_jobs", type=int, default=1,
        help="Number of parallel worker processes for gene trend computation. "
             "Use -1 to use all available CPUs.")
    parser.add_argument(
        "--random_state", type=int, default=0,
        help="Random seed for reproducibility (PCA, UMAP, diffusion map).")

    return parser


def compute_diffusion_pseudotime(adata, n_neighbors=15, n_pcs=30,
                                 random_state=0):
    """Compute diffusion pseudotime from gene expression.

    Pipeline: normalize -> HVG -> PCA -> neighbors -> diffusion map ->
    DPT. The root cell is automatically selected as the cell with
    highest connectivity (degree) in the neighborhood graph.

    :param adata: AnnData with raw gene counts.
    :param n_neighbors: number of neighbors for KNN graph.
    :param n_pcs: number of principal components.
    :param random_state: random seed for PCA and neighbors.
    :returns: adata with dpt_pseudotime in obs.
    """
    import scanpy as sc

    logger = get_named_logger("DPT")
    logger.info("Computing diffusion pseudotime.")

    # Work on a copy to avoid modifying the original
    adata = adata.copy()

    # Normalize
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # HVG selection
    sc.pp.highly_variable_genes(
        adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata_hvg = adata[:, adata.var.highly_variable].copy()

    # Scale
    sc.pp.scale(adata_hvg, max_value=10)

    # PCA
    n_pcs = min(n_pcs, adata_hvg.shape[0] - 1, adata_hvg.shape[1] - 1)
    logger.info(f"Running PCA with {n_pcs} components.")
    sc.tl.pca(adata_hvg, n_comps=n_pcs, random_state=random_state)

    # Neighbors
    sc.pp.neighbors(
        adata_hvg, n_neighbors=n_neighbors, n_pcs=n_pcs,
        random_state=random_state)

    # Diffusion map
    logger.info("Computing diffusion map.")
    sc.tl.diffmap(adata_hvg)

    # Auto-select root cell: highest connectivity in the graph
    try:
        connectivities = adata_hvg.obsp['connectivities']
        if scipy.sparse.issparse(connectivities):
            degrees = np.asarray(connectivities.sum(axis=1)).ravel()
        else:
            degrees = np.sum(connectivities, axis=1).ravel()
        root_idx = int(np.argmax(degrees))
    except (KeyError, AttributeError):
        root_idx = 0

    adata_hvg.uns['iroot'] = root_idx
    logger.info(f"Root cell index: {root_idx}.")

    # DPT
    sc.tl.dpt(adata_hvg)

    # Transfer pseudotime back to original adata
    adata.obs['dpt_pseudotime'] = adata_hvg.obs['dpt_pseudotime'].values

    # Handle infinite values (disconnected components)
    inf_mask = np.isinf(adata.obs['dpt_pseudotime'])
    if inf_mask.any():
        max_finite = adata.obs.loc[
            ~inf_mask, 'dpt_pseudotime'].max()
        adata.obs.loc[inf_mask, 'dpt_pseudotime'] = max_finite
        logger.info(
            f"Replaced {int(inf_mask.sum())} infinite pseudotime values.")

    # Transfer diffmap embedding
    if 'X_diffmap' in adata_hvg.obsm:
        adata.obsm['X_diffmap'] = adata_hvg.obsm['X_diffmap']

    logger.info("Diffusion pseudotime computation complete.")
    return adata


def _compute_gene_trends(
        gene, transcripts, gene_matrix, bin_labels, bin_midpoints, n_bins,
        tx_names_list):
    """Compute pseudotime trend statistics for all transcripts of one gene.

    Intended to be called in a worker process via ProcessPoolExecutor.
    Accepts a pre-extracted dense gene matrix to avoid pickling sparse
    AnnData objects.

    :param gene: gene identifier string.
    :param transcripts: list of transcript name strings for this gene.
    :param gene_matrix: 2D numpy array (n_cells x n_transcripts), dense,
        columns ordered to match ``transcripts``.
    :param bin_labels: 1D int array mapping each cell to a pseudotime bin.
    :param bin_midpoints: 1D float array of bin midpoint pseudotime values.
    :param n_bins: total number of bins.
    :param tx_names_list: unused (kept for API consistency); pass None.
    :returns: list of trend dicts (may be empty).
    """
    trend_rows = []
    n_tx = len(transcripts)
    bin_proportions = np.zeros((n_bins, n_tx))

    for b in range(n_bins):
        mask = bin_labels == b
        if mask.sum() == 0:
            continue
        bin_expr = gene_matrix[mask]
        bin_total = bin_expr.sum()
        if bin_total > 0:
            bin_proportions[b] = bin_expr.sum(axis=0)
            row_total = bin_proportions[b].sum()
            if row_total > 0:
                bin_proportions[b] /= row_total

    for j, tx in enumerate(transcripts):
        props = bin_proportions[:, j]
        if np.std(props) == 0:
            continue
        r, pval = scipy.stats.spearmanr(bin_midpoints, props)
        if np.isnan(r):
            continue
        if r > 0.3:
            trend = 'increasing'
        elif r < -0.3:
            trend = 'decreasing'
        else:
            trend = 'stable'
        trend_rows.append({
            'gene': gene,
            'transcript': tx,
            'spearman_r': round(r, 4),
            'pvalue': pval,
            'trend': trend,
            'mean_proportion': round(float(props.mean()), 4),
        })

    return trend_rows


def compute_isoform_trends(adata_tx, gene_map, pseudotime, n_bins=10,
                           n_jobs=1):
    """Compute isoform proportion trends along pseudotime.

    Bins cells by pseudotime and computes mean isoform proportions
    within each bin for each gene. Assesses monotonic trends using
    Spearman correlation of isoform proportion vs bin midpoint.

    :param adata_tx: AnnData with transcript-level counts.
    :param gene_map: DataFrame with transcript_id, gene_id columns.
    :param pseudotime: Series mapping barcode -> pseudotime value.
    :param n_bins: number of pseudotime bins.
    :param n_jobs: number of parallel worker processes. 1 = sequential.
        Pass -1 to use all available CPUs.
    :returns: DataFrame with gene, transcript, spearman_r, pvalue, trend.
    """
    logger = get_named_logger("Trends")
    logger.info(f"Computing isoform trends in {n_bins} pseudotime bins.")

    # Align barcodes
    common = sorted(
        set(adata_tx.obs_names) & set(pseudotime.index))
    if len(common) == 0:
        logger.warning("No overlapping barcodes between matrix and pseudotime.")
        return pd.DataFrame()

    adata_sub = adata_tx[common].copy()
    pt_values = pseudotime.loc[common].values.astype(float)

    # Remove cells with NaN pseudotime
    valid = ~np.isnan(pt_values)
    if valid.sum() < n_bins:
        logger.warning("Too few cells with valid pseudotime.")
        return pd.DataFrame()

    adata_sub = adata_sub[valid].copy()
    pt_values = pt_values[valid]

    # Bin cells by pseudotime
    bin_edges = np.linspace(pt_values.min(), pt_values.max(), n_bins + 1)
    bin_labels = np.clip(
        np.digitize(pt_values, bin_edges) - 1, 0, n_bins - 1)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Build gene -> transcript mapping
    tx_to_gene = dict(zip(
        gene_map.iloc[:, 0].astype(str),
        gene_map.iloc[:, 1].astype(str)))

    # Map transcripts in matrix to genes
    tx_names = list(adata_sub.var_names)
    tx_gene_map = {}
    for tx in tx_names:
        gene = tx_to_gene.get(tx)
        if gene:
            tx_gene_map.setdefault(gene, []).append(tx)

    X = adata_sub.X

    # Pre-extract dense per-gene arrays so workers receive picklable objects
    gene_items = []
    for gene, transcripts in tx_gene_map.items():
        if len(transcripts) < 2:
            continue
        tx_indices = [tx_names.index(tx) for tx in transcripts]
        if scipy.sparse.issparse(X):
            gene_mat = safe_toarray(
                X[:, tx_indices], context=f"gene {gene} transcript block")
        else:
            gene_mat = np.asarray(X[:, tx_indices])
        gene_items.append((gene, transcripts, gene_mat))

    if n_jobs == -1:
        import os
        n_jobs = os.cpu_count() or 1

    trend_rows = []

    if n_jobs == 1:
        for gene, transcripts, gene_mat in gene_items:
            rows = _compute_gene_trends(
                gene, transcripts, gene_mat,
                bin_labels, bin_midpoints, n_bins, None)
            trend_rows.extend(rows)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    _compute_gene_trends,
                    gene, transcripts, gene_mat,
                    bin_labels, bin_midpoints, n_bins, None)
                for gene, transcripts, gene_mat in gene_items
            ]
            for f in futures:
                trend_rows.extend(f.result())

    trends_df = pd.DataFrame(trend_rows)
    if len(trends_df) > 0:
        # BH correction
        trends_df['pvalue_adj'] = _bh_correct(trends_df['pvalue'].values)
        trends_df = trends_df.sort_values(
            'pvalue_adj').reset_index(drop=True)

    n_dynamic = int(
        (trends_df['trend'] != 'stable').sum()) if len(trends_df) > 0 else 0
    logger.info(
        f"Isoform trends: {len(trends_df)} transcripts analyzed, "
        f"{n_dynamic} with dynamic trends.")
    return trends_df


def detect_trajectory_switching(trends_df, pval_threshold=0.05):
    """Detect isoform switching events along the trajectory.

    For genes with at least one increasing and one decreasing isoform
    (both significant), identifies the switching point as the pseudotime
    bin where the dominant isoform changes.

    :param trends_df: DataFrame from compute_isoform_trends.
    :param pval_threshold: adjusted p-value threshold for significance.
    :returns: DataFrame with switching events.
    """
    logger = get_named_logger("Switching")

    if len(trends_df) == 0:
        return pd.DataFrame()

    sig = trends_df[trends_df['pvalue_adj'] <= pval_threshold].copy()
    if len(sig) == 0:
        logger.info("No significant trends; no switching detected.")
        return pd.DataFrame()

    switching_rows = []

    for gene, grp in sig.groupby('gene'):
        increasing = grp[grp['trend'] == 'increasing']
        decreasing = grp[grp['trend'] == 'decreasing']

        if len(increasing) == 0 or len(decreasing) == 0:
            continue

        # Each pair of (increasing, decreasing) is a potential switch
        for _, inc_row in increasing.iterrows():
            for _, dec_row in decreasing.iterrows():
                # Estimate switching point from correlation magnitudes
                # Stronger anticorrelation = sharper switch
                r_inc = inc_row['spearman_r']
                r_dec = dec_row['spearman_r']
                switch_strength = abs(r_inc) + abs(r_dec)

                switching_rows.append({
                    'gene': gene,
                    'transcript_increasing': inc_row['transcript'],
                    'transcript_decreasing': dec_row['transcript'],
                    'r_increasing': round(r_inc, 4),
                    'r_decreasing': round(r_dec, 4),
                    'pvalue_increasing': inc_row['pvalue_adj'],
                    'pvalue_decreasing': dec_row['pvalue_adj'],
                    'switch_strength': round(switch_strength, 4),
                })

    switching_df = pd.DataFrame(switching_rows)
    if len(switching_df) > 0:
        switching_df = switching_df.sort_values(
            'switch_strength', ascending=False).reset_index(drop=True)

    logger.info(
        f"Detected {len(switching_df)} potential switching events "
        f"in {switching_df['gene'].nunique() if len(switching_df) > 0 else 0}"
        f" genes.")
    return switching_df


from ._stats import bh_correct as _bh_correct, safe_toarray


def main(args):
    """Run isoform-aware trajectory analysis pipeline."""
    import scanpy as sc

    logger = get_named_logger("Trajectory")
    logger.info("Starting isoform-aware trajectory analysis.")

    # Load gene expression matrix
    logger.info(f"Loading gene matrix from {args.gene_matrix_dir}.")
    adata_gene = sc.read_10x_mtx(
        str(args.gene_matrix_dir), var_names='gene_symbols')
    adata_gene.var_names_make_unique()
    logger.info(
        f"Gene matrix: {adata_gene.shape[0]} cells x "
        f"{adata_gene.shape[1]} genes.")

    # Load transcript expression matrix
    logger.info(
        f"Loading transcript matrix from {args.transcript_matrix_dir}.")
    adata_tx = sc.read_10x_mtx(
        str(args.transcript_matrix_dir), var_names='gene_symbols')
    adata_tx.var_names_make_unique()
    logger.info(
        f"Transcript matrix: {adata_tx.shape[0]} cells x "
        f"{adata_tx.shape[1]} transcripts.")

    # Load gene-transcript mapping
    gene_map = pd.read_csv(args.gene_transcript_map, sep='\t')
    logger.info(
        f"Loaded {len(gene_map)} transcript-gene mappings.")

    # Compute diffusion pseudotime from gene expression
    adata_gene = compute_diffusion_pseudotime(
        adata_gene,
        n_neighbors=args.n_dpt_neighbors,
        n_pcs=args.n_pcs,
        random_state=args.random_state)

    # Write pseudotime
    pseudotime_df = pd.DataFrame({
        'barcode': adata_gene.obs_names,
        'dpt_pseudotime': adata_gene.obs['dpt_pseudotime'].values,
    })
    pseudotime_df.to_csv(args.output_pseudotime, sep='\t', index=False)
    logger.info(
        f"Pseudotime written to {args.output_pseudotime} "
        f"({len(pseudotime_df)} cells).")

    # Create pseudotime Series for trend analysis
    pseudotime = pd.Series(
        adata_gene.obs['dpt_pseudotime'].values,
        index=adata_gene.obs_names)

    # Compute isoform trends along pseudotime
    trends_df = compute_isoform_trends(
        adata_tx, gene_map, pseudotime, n_bins=args.n_bins,
        n_jobs=getattr(args, 'n_jobs', 1))

    if len(trends_df) > 0:
        trends_df.to_csv(
            args.output_isoform_dynamics, sep='\t', index=False)
        logger.info(
            f"Isoform dynamics written to "
            f"{args.output_isoform_dynamics}.")
    else:
        pd.DataFrame().to_csv(
            args.output_isoform_dynamics, sep='\t', index=False)

    # Detect switching events
    switching_df = detect_trajectory_switching(trends_df)
    if len(switching_df) > 0:
        switching_df.to_csv(
            args.output_switching_trajectory, sep='\t', index=False)
        logger.info(
            f"Switching events written to "
            f"{args.output_switching_trajectory}.")
    else:
        pd.DataFrame().to_csv(
            args.output_switching_trajectory, sep='\t', index=False)

    # Summary
    n_genes_analyzed = trends_df['gene'].nunique() \
        if len(trends_df) > 0 else 0
    n_dynamic = int(
        (trends_df['trend'] != 'stable').sum()) if len(trends_df) > 0 else 0
    n_switching_genes = switching_df['gene'].nunique() \
        if len(switching_df) > 0 else 0

    summary = {
        'n_cells': int(adata_gene.shape[0]),
        'n_genes': int(adata_gene.shape[1]),
        'n_transcripts': int(adata_tx.shape[1]),
        'n_pcs': args.n_pcs,
        'n_dpt_neighbors': args.n_dpt_neighbors,
        'n_bins': args.n_bins,
        'n_genes_analyzed': n_genes_analyzed,
        'n_dynamic_isoforms': n_dynamic,
        'n_switching_events': len(switching_df),
        'n_switching_genes': n_switching_genes,
        'pseudotime_range': [
            float(pseudotime.min()),
            float(pseudotime.max()),
        ],
    }

    if len(switching_df) > 0:
        summary['top_switching_genes'] = switching_df.head(20)[[
            'gene', 'transcript_increasing', 'transcript_decreasing',
            'switch_strength',
        ]].to_dict('records')

    with open(args.output_summary, 'w') as fh:
        json.dump(summary, fh, indent=2)

    logger.info(
        f"Trajectory analysis complete: {n_genes_analyzed} genes analyzed, "
        f"{n_dynamic} dynamic isoforms, {n_switching_genes} switching genes.")
