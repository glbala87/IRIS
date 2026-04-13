"""Cluster analysis using Scanpy and Seurat-inspired methods.

Provides Leiden/Louvain clustering, neighborhood graph construction,
marker gene detection (Scanpy rank_genes_groups), and SCTransform-style
normalization (Seurat).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse

from ._logging import get_named_logger, wf_parser


def argparser():
    """Create argument parser."""
    parser = wf_parser("cluster_analysis")

    parser.add_argument(
        "matrix_dir", type=Path,
        help="Path to processed MEX matrix directory.")
    parser.add_argument(
        "--output_clusters", type=Path, default="clusters.tsv",
        help="Output TSV with cluster assignments per cell.")
    parser.add_argument(
        "--output_markers", type=Path, default="marker_genes.tsv",
        help="Output TSV with marker genes per cluster.")
    parser.add_argument(
        "--output_umap", type=Path, default="cluster.umap.tsv",
        help="Output UMAP coordinates with cluster labels.")
    parser.add_argument(
        "--output_summary", type=Path, default="cluster_summary.json",
        help="Output JSON with clustering summary statistics.")

    grp = parser.add_argument_group("Scanpy clustering")
    grp.add_argument(
        "--cluster_method", default="leiden",
        choices=["leiden", "louvain"],
        help="Clustering algorithm (Scanpy).")
    grp.add_argument(
        "--resolution", type=float, default=1.0,
        help="Resolution parameter for Leiden/Louvain clustering.")
    grp.add_argument(
        "--n_neighbors", type=int, default=15,
        help="Number of neighbors for KNN graph (Scanpy/Seurat).")
    grp.add_argument(
        "--n_pcs", type=int, default=50,
        help="Number of PCs for neighborhood graph.")

    grp = parser.add_argument_group("Seurat-style normalization")
    grp.add_argument(
        "--normalization", default="scanpy",
        choices=["scanpy", "sctransform"],
        help="Normalization method: scanpy (log-normalize) or "
             "sctransform (Seurat v3 SCTransform-inspired).")
    grp.add_argument(
        "--norm_count", type=int, default=10000,
        help="Target counts for scanpy normalization.")

    grp = parser.add_argument_group("Marker gene detection")
    grp.add_argument(
        "--n_marker_genes", type=int, default=25,
        help="Number of top marker genes per cluster.")
    grp.add_argument(
        "--marker_method", default="wilcoxon",
        choices=["wilcoxon", "t-test", "t-test_overestim_var", "logreg"],
        help="Method for marker gene detection (Scanpy rank_genes_groups).")

    grp = parser.add_argument_group("Cell Ranger cell filtering")
    grp.add_argument(
        "--cellranger_cell_calling", action="store_true",
        help="Use Cell Ranger-style EmptyDrops cell calling.")
    grp.add_argument(
        "--expected_cells", type=int, default=3000,
        help="Expected number of cells (Cell Ranger).")
    grp.add_argument(
        "--emptydrops_n_sims", type=int, default=10000,
        help="Number of Monte Carlo simulations for EmptyDrops cell calling.")

    grp = parser.add_argument_group("UMAP")
    grp.add_argument(
        "--min_dist", type=float, default=0.3,
        help="UMAP min_dist (Scanpy default).")
    grp.add_argument(
        "--spread", type=float, default=1.0,
        help="UMAP spread parameter.")

    parser.add_argument(
        "--random_state", type=int, default=0,
        help="Random seed for reproducibility (PCA, UMAP, clustering).")

    return parser


def load_mex_to_anndata(matrix_dir):
    """Load MEX format matrix into an AnnData object.

    Follows Cell Ranger MEX conventions: matrix.mtx.gz, barcodes.tsv.gz,
    features.tsv.gz.
    """
    import scanpy as sc

    adata = sc.read_10x_mtx(str(matrix_dir), var_names='gene_symbols')
    adata.var_names_make_unique()
    return adata


def sctransform_normalize(adata):
    """SCTransform-inspired normalization (Seurat v3 concept).

    Implements a simplified version of the SCTransform variance-stabilizing
    transformation: fits a regularized negative binomial model per gene,
    then returns Pearson residuals as normalized expression values.

    Reference: Hafemeister & Satija, Genome Biology 2019.
    """
    import scanpy as sc

    logger = get_named_logger("SCTransform")
    logger.info("Performing SCTransform-style normalization.")

    # Store raw counts
    adata.layers['counts'] = adata.X.copy()

    # SCTransform uses Pearson residuals from a regularized NB regression
    # scanpy provides an efficient implementation of this concept
    sc.experimental.pp.normalize_pearson_residuals(adata)

    return adata


def scanpy_normalize(adata, target_sum=10000):
    """Standard Scanpy/Seurat log-normalization.

    Equivalent to Seurat NormalizeData(normalization.method = "LogNormalize").
    """
    import scanpy as sc

    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata


def cellranger_cell_calling(adata, expected_cells=3000, random_state=0, n_sims=10000):
    """Cell Ranger-style EmptyDrops cell calling.

    Implements the two-step approach from Cell Ranger:
    1. Top N barcodes by UMI count are called as cells
    2. EmptyDrops-like test on remaining barcodes using a Dirichlet-multinomial
       model to distinguish cells from ambient RNA.

    Reference: Lun et al., Genome Biology 2019.
    """
    logger = get_named_logger("CellCalling")

    total_counts = np.array(adata.X.sum(axis=1)).flatten()
    sorted_counts = np.sort(total_counts)[::-1]

    # Step 1: robust initial cell estimate (Cell Ranger ordmag algorithm)
    n_top = max(1, int(expected_cells * 0.01))
    top_counts = sorted_counts[:n_top]
    ordmag_threshold = np.median(top_counts) / 10

    # Step 2: EmptyDrops-like significance testing
    # Estimate ambient profile from low-count barcodes
    ambient_threshold = max(100, ordmag_threshold * 0.1)
    ambient_mask = total_counts < ambient_threshold
    if ambient_mask.sum() > 0:
        ambient_profile = np.array(
            adata.X[ambient_mask].sum(axis=0)).flatten()
        ambient_profile = ambient_profile / ambient_profile.sum()
    else:
        ambient_profile = None

    # Cells are those above the threshold or passing significance test
    cell_mask = total_counts >= ordmag_threshold

    # Monte Carlo test for marginal barcodes (simplified EmptyDrops)
    marginal_mask = (
        (total_counts < ordmag_threshold) &
        (total_counts >= ambient_threshold))
    if ambient_profile is not None and marginal_mask.sum() > 0:
        # n_sims controlled by caller via parameter
        marginal_indices = np.where(marginal_mask)[0]
        for idx in marginal_indices:
            obs_counts = np.array(adata.X[idx].todense()).flatten() \
                if scipy.sparse.issparse(adata.X) \
                else adata.X[idx]
            obs_total = obs_counts.sum()
            if obs_total == 0:
                continue
            # Log-likelihood ratio vs ambient
            obs_profile = obs_counts / obs_total
            nonzero = obs_profile > 0
            if nonzero.sum() < 2:
                continue
            obs_llr = np.sum(
                obs_counts[nonzero] * np.log(
                    obs_profile[nonzero] / (ambient_profile[nonzero] + 1e-10)))
            # Simulate from ambient
            rng = np.random.default_rng(random_state)
            sim_llrs = np.zeros(n_sims)
            for s in range(n_sims):
                sim = rng.multinomial(int(obs_total), ambient_profile)
                sim_profile = sim / obs_total
                nz = sim_profile > 0
                sim_llrs[s] = np.sum(
                    sim[nz] * np.log(
                        sim_profile[nz] / (ambient_profile[nz] + 1e-10)))
            pval = (np.sum(sim_llrs >= obs_llr) + 1) / (n_sims + 1)
            if pval < 0.01:
                cell_mask[idx] = True

    n_called = cell_mask.sum()
    logger.info(
        f"Cell Ranger-style calling: {n_called} cells from "
        f"{len(total_counts)} barcodes "
        f"(threshold={ordmag_threshold:.0f} UMIs).")

    adata = adata[cell_mask].copy()
    return adata


def run_clustering(adata, args):
    """Run Scanpy clustering pipeline.

    Follows Scanpy best practices and parallels the Seurat clustering workflow:
    1. HVG selection (analogous to Seurat FindVariableFeatures)
    2. PCA (analogous to Seurat RunPCA)
    3. KNN graph (analogous to Seurat FindNeighbors)
    4. Leiden/Louvain clustering (analogous to Seurat FindClusters)
    5. UMAP embedding (analogous to Seurat RunUMAP)
    """
    import scanpy as sc

    logger = get_named_logger("Clustering")

    # Highly variable genes (Seurat-style)
    logger.info("Selecting highly variable genes.")
    sc.pp.highly_variable_genes(
        adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable].copy()

    # Scale (Seurat ScaleData) — scanpy >= 1.9 handles sparse natively
    sc.pp.scale(adata, max_value=10)

    # PCA (Seurat RunPCA)
    n_pcs = min(args.n_pcs, adata.shape[0] - 1, adata.shape[1] - 1)
    logger.info(f"Running PCA with {n_pcs} components.")
    sc.tl.pca(adata, n_comps=n_pcs, random_state=args.random_state)

    # Neighborhood graph (Seurat FindNeighbors)
    logger.info(
        f"Building neighborhood graph: "
        f"n_neighbors={args.n_neighbors}, n_pcs={n_pcs}.")
    sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, n_pcs=n_pcs)

    # Clustering (Seurat FindClusters uses Louvain/Leiden)
    logger.info(
        f"Clustering with {args.cluster_method} "
        f"(resolution={args.resolution}).")
    if args.cluster_method == "leiden":
        sc.tl.leiden(
            adata, resolution=args.resolution, key_added='cluster',
            flavor='igraph', n_iterations=2, directed=False,
            random_state=args.random_state)
    else:
        sc.tl.louvain(
            adata, resolution=args.resolution, key_added='cluster',
            flavor='igraph', random_state=args.random_state)

    # UMAP (Seurat RunUMAP)
    logger.info("Computing UMAP embedding.")
    sc.tl.umap(
        adata, min_dist=args.min_dist, spread=args.spread,
        random_state=args.random_state)

    return adata


def find_marker_genes(adata, method='wilcoxon', n_genes=25):
    """Find marker genes per cluster.

    Uses Scanpy rank_genes_groups which implements:
    - Wilcoxon rank-sum test (default, robust)
    - t-test (Seurat-style)
    - t-test with overestimated variance
    - Logistic regression

    Analogous to Seurat FindAllMarkers.
    """
    import scanpy as sc

    logger = get_named_logger("MarkerGenes")
    logger.info(f"Finding marker genes with method={method}.")

    use_raw = adata.raw is not None
    sc.tl.rank_genes_groups(
        adata, groupby='cluster', method=method,
        n_genes=n_genes, use_raw=use_raw)

    # Extract results into a DataFrame
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    rows = []
    for group in groups:
        for i in range(n_genes):
            try:
                rows.append({
                    'cluster': group,
                    'gene': result['names'][group][i],
                    'score': float(result['scores'][group][i]),
                    'pval': float(result['pvals'][group][i]),
                    'pval_adj': float(result['pvals_adj'][group][i]),
                    'log2fc': float(result['logfoldchanges'][group][i]),
                })
            except (IndexError, KeyError):
                break

    markers_df = pd.DataFrame(rows)
    return markers_df


def main(args):
    """Run cluster analysis pipeline."""
    logger = get_named_logger("ClusterAnalysis")
    logger.info("Starting cluster analysis.")

    # Load data
    logger.info(f"Loading matrix from {args.matrix_dir}.")
    adata = load_mex_to_anndata(args.matrix_dir)
    logger.info(
        f"Loaded matrix: {adata.shape[0]} cells x {adata.shape[1]} genes.")

    # Cell Ranger-style cell calling (optional)
    if args.cellranger_cell_calling:
        adata = cellranger_cell_calling(
            adata, args.expected_cells, args.random_state,
            n_sims=args.emptydrops_n_sims)
        logger.info(f"After cell calling: {adata.shape[0]} cells.")

    # QC metrics (Scanpy)
    import scanpy as sc
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # Normalization
    if args.normalization == "sctransform":
        adata = sctransform_normalize(adata)
    else:
        adata = scanpy_normalize(adata, target_sum=args.norm_count)

    # Clustering
    adata = run_clustering(adata, args)

    # Write cluster assignments
    clusters_df = pd.DataFrame({
        'barcode': adata.obs_names,
        'cluster': adata.obs['cluster'].values
    })
    clusters_df.to_csv(args.output_clusters, sep='\t', index=False)
    logger.info(
        f"Identified {adata.obs['cluster'].nunique()} clusters "
        f"across {adata.shape[0]} cells.")

    # Find marker genes
    markers_df = find_marker_genes(
        adata, method=args.marker_method, n_genes=args.n_marker_genes)
    markers_df.to_csv(args.output_markers, sep='\t', index=False)
    logger.info(f"Found markers for {markers_df['cluster'].nunique()} clusters.")

    # Write UMAP with cluster labels
    umap_coords = adata.obsm['X_umap']
    umap_df = pd.DataFrame(
        umap_coords, columns=['D1', 'D2'], index=adata.obs_names)
    umap_df['cluster'] = adata.obs['cluster'].values
    umap_df.to_csv(args.output_umap, sep='\t', index_label='CB')

    # Cluster summary statistics
    summary = {
        'n_cells': int(adata.shape[0]),
        'n_clusters': int(adata.obs['cluster'].nunique()),
        'cluster_method': args.cluster_method,
        'resolution': args.resolution,
        'normalization': args.normalization,
        'cells_per_cluster': adata.obs['cluster'].value_counts().to_dict(),
        'n_hvg': int(adata.shape[1]),
    }
    with open(args.output_summary, 'w') as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Cluster analysis complete.")
