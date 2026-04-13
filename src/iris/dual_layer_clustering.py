"""Dual-layer clustering using gene expression AND isoform usage patterns.

Module 3 of the isosceles pipeline. Clusters cells at two levels —
gene expression and isoform usage proportions — then computes a joint
embedding that captures both transcriptional identity and splicing
diversity. This enables discovery of cell states that differ not in
which genes they express, but in how those genes are spliced.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse

from ._logging import get_named_logger, wf_parser
from ._stats import safe_toarray


def argparser():
    """Create argument parser."""
    parser = wf_parser("dual_layer_clustering")

    parser.add_argument(
        "gene_matrix_dir", type=Path,
        help="Path to processed gene MEX matrix directory.")
    parser.add_argument(
        "transcript_matrix_dir", type=Path,
        help="Path to processed transcript MEX matrix directory.")

    parser.add_argument(
        "--gene_transcript_map", type=Path, required=True,
        help="TSV mapping transcript_id to gene_id (two columns).")
    parser.add_argument(
        "--output_gene_clusters", type=Path, default="gene_clusters.tsv",
        help="Output TSV with gene-level cluster assignments.")
    parser.add_argument(
        "--output_isoform_clusters", type=Path,
        default="isoform_clusters.tsv",
        help="Output TSV with isoform-usage cluster assignments.")
    parser.add_argument(
        "--output_joint_clusters", type=Path,
        default="joint_clusters.tsv",
        help="Output TSV with joint cluster assignments.")
    parser.add_argument(
        "--output_joint_umap", type=Path, default="joint.umap.tsv",
        help="Output UMAP coordinates from joint embedding.")
    parser.add_argument(
        "--output_diversity", type=Path,
        default="isoform_diversity.tsv",
        help="Output TSV with per-cell isoform diversity indices.")
    parser.add_argument(
        "--output_comparison", type=Path,
        default="cluster_comparison.json",
        help="Output JSON comparing gene vs isoform clusterings.")

    grp = parser.add_argument_group("Clustering parameters")
    grp.add_argument(
        "--cluster_method", default="leiden",
        choices=["leiden", "louvain"],
        help="Clustering algorithm.")
    grp.add_argument(
        "--resolution", type=float, default=1.0,
        help="Resolution for gene-level clustering.")
    grp.add_argument(
        "--isoform_resolution", type=float, default=1.0,
        help="Resolution for isoform-usage clustering.")
    grp.add_argument(
        "--n_neighbors", type=int, default=15,
        help="Number of neighbors for KNN graph.")
    grp.add_argument(
        "--n_pcs", type=int, default=50,
        help="Number of principal components.")

    grp = parser.add_argument_group("Isoform parameters")
    grp.add_argument(
        "--min_isoforms_per_gene", type=int, default=2,
        help="Minimum isoforms per gene to include in usage matrix.")
    grp.add_argument(
        "--diversity_metric", default="shannon",
        choices=["shannon", "simpson"],
        help="Diversity metric for isoform usage.")

    grp = parser.add_argument_group("Joint embedding parameters")
    grp.add_argument(
        "--isoform_weight", type=float, default=3.0,
        help="Weight for isoform graph in joint embedding relative to "
             "gene graph. Values > 1 upweight isoform signal. At 3.0, "
             "the joint graph is 75%% isoform / 25%% gene. Increase for "
             "datasets where isoform usage drives cell state; decrease "
             "when gene expression is more informative.")
    grp.add_argument(
        "--normalize_pcs", action='store_true', default=True,
        help="L2-normalize each PCA space before concatenation "
             "to prevent variance imbalance.")
    grp.add_argument(
        "--no_normalize_pcs", dest='normalize_pcs', action='store_false',
        help="Disable PCA normalization before concatenation.")

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


def load_gene_transcript_map(map_file):
    """Load a two-column TSV mapping transcript_id to gene_id.

    :param map_file: Path to TSV with columns transcript_id, gene_id.
    :returns: dict mapping transcript_id -> gene_id.
    """
    logger = get_named_logger("GeneMap")
    df = pd.read_csv(map_file, sep='\t', header=0)
    cols = df.columns.tolist()
    if len(cols) < 2:
        raise ValueError(
            f"Gene-transcript map must have at least 2 columns, "
            f"got {len(cols)}: {cols}")
    transcript_col, gene_col = cols[0], cols[1]
    gene_map = dict(zip(df[transcript_col], df[gene_col]))
    logger.info(
        f"Loaded {len(gene_map)} transcript-to-gene mappings "
        f"from {map_file}.")
    return gene_map


def compute_isoform_usage(adata_transcript, gene_map, min_isoforms=2):
    """Compute isoform usage proportions from transcript counts.

    For each cell, for each gene with at least `min_isoforms` detected
    transcripts, compute the proportion of each transcript relative to
    the total count of all transcripts of that gene in that cell.

    Operates on sparse matrices gene-by-gene to avoid dense explosion.

    :param adata_transcript: AnnData with transcript-level counts.
    :param gene_map: dict mapping transcript_id -> gene_id.
    :param min_isoforms: minimum transcripts per gene to include.
    :returns: AnnData with usage proportions (cells x transcripts).
    """
    import anndata

    logger = get_named_logger("IsoUsage")

    transcript_ids = adata_transcript.var_names.tolist()
    n_cells = adata_transcript.shape[0]

    # Group transcript indices by gene
    gene_to_tx_indices = {}
    for idx, tx_id in enumerate(transcript_ids):
        gene_id = gene_map.get(tx_id)
        if gene_id is None:
            continue
        gene_to_tx_indices.setdefault(gene_id, []).append(idx)

    # Keep only genes with >= min_isoforms transcripts
    multi_genes = {
        g: indices for g, indices in gene_to_tx_indices.items()
        if len(indices) >= min_isoforms}
    logger.info(
        f"Found {len(multi_genes)} genes with >= {min_isoforms} "
        f"isoforms for usage computation.")

    if len(multi_genes) == 0:
        logger.warning("No multi-isoform genes found. Returning empty matrix.")
        empty_X = scipy.sparse.csr_matrix((n_cells, 0))
        adata_usage = anndata.AnnData(
            X=empty_X, obs=adata_transcript.obs.copy())
        return adata_usage

    # Collect all transcript indices that belong to multi-isoform genes
    kept_tx_indices = []
    tx_to_gene_for_kept = {}
    for gene_id, indices in sorted(multi_genes.items()):
        for idx in indices:
            kept_tx_indices.append(idx)
            tx_to_gene_for_kept[idx] = gene_id

    kept_tx_indices_sorted = sorted(kept_tx_indices)
    kept_tx_names = [transcript_ids[i] for i in kept_tx_indices_sorted]

    # Build the usage matrix gene-by-gene using COO for memory efficiency
    X = adata_transcript.X
    is_sparse = scipy.sparse.issparse(X)

    # Map from original index to position in output matrix
    orig_to_new = {
        orig_idx: new_idx
        for new_idx, orig_idx in enumerate(kept_tx_indices_sorted)}

    # Accumulate COO triplets (much more memory-efficient than lil_matrix)
    coo_rows = []
    coo_cols = []
    coo_vals = []

    for gene_id, indices in multi_genes.items():
        # Extract columns for this gene's transcripts
        if is_sparse:
            gene_block = safe_toarray(
                X[:, indices], context=f"isoform usage block for gene {gene_id}")
        else:
            gene_block = np.asarray(X[:, indices])

        # Per-cell total for this gene
        gene_totals = gene_block.sum(axis=1, keepdims=True)
        mask = gene_totals.ravel() > 0
        if mask.sum() == 0:
            continue

        proportions = np.zeros_like(gene_block, dtype=np.float64)
        proportions[mask] = gene_block[mask] / gene_totals[mask]

        for local_j, orig_idx in enumerate(indices):
            new_idx = orig_to_new[orig_idx]
            col = proportions[:, local_j]
            nz = np.where(col > 0)[0]
            if len(nz) > 0:
                coo_rows.append(nz)
                coo_cols.append(np.full(len(nz), new_idx, dtype=int))
                coo_vals.append(col[nz])

    if coo_rows:
        all_rows = np.concatenate(coo_rows)
        all_cols = np.concatenate(coo_cols)
        all_vals = np.concatenate(coo_vals)
        usage_csr = scipy.sparse.coo_matrix(
            (all_vals, (all_rows, all_cols)),
            shape=(n_cells, len(kept_tx_indices_sorted))
        ).tocsr()
    else:
        usage_csr = scipy.sparse.csr_matrix(
            (n_cells, len(kept_tx_indices_sorted)))

    var_df = pd.DataFrame(index=kept_tx_names)
    var_df['gene_id'] = [
        tx_to_gene_for_kept[kept_tx_indices_sorted[i]]
        for i in range(len(kept_tx_names))]

    adata_usage = anndata.AnnData(
        X=usage_csr,
        obs=adata_transcript.obs.copy(),
        var=var_df)
    adata_usage.var_names_make_unique()

    logger.info(
        f"Isoform usage matrix: {adata_usage.shape[0]} cells x "
        f"{adata_usage.shape[1]} transcripts.")
    return adata_usage


def compute_diversity_index(adata_usage, gene_map, metric='shannon'):
    """Compute per-cell isoform diversity index.

    For each cell, for each multi-isoform gene, compute a diversity
    metric on the isoform usage proportions, then average across genes.

    :param adata_usage: AnnData with isoform usage proportions.
    :param gene_map: dict transcript_id -> gene_id.
    :param metric: 'shannon' for Shannon entropy or 'simpson' for
        Simpson's diversity index.
    :returns: DataFrame with columns barcode, diversity_index,
        n_genes_multi_isoform.
    """
    logger = get_named_logger("Diversity")
    logger.info(f"Computing {metric} diversity index per cell.")

    transcript_ids = adata_usage.var_names.tolist()
    n_cells = adata_usage.shape[0]

    # Group transcript indices by gene
    gene_to_tx_indices = {}
    for idx, tx_id in enumerate(transcript_ids):
        gene_id = gene_map.get(tx_id)
        if gene_id is None:
            # Fall back to gene_id stored in var if available
            if 'gene_id' in adata_usage.var.columns:
                gene_id = adata_usage.var.iloc[idx]['gene_id']
        if gene_id is not None:
            gene_to_tx_indices.setdefault(gene_id, []).append(idx)

    X = adata_usage.X
    is_sparse = scipy.sparse.issparse(X)

    diversity_scores = np.zeros(n_cells, dtype=np.float64)
    gene_counts = np.zeros(n_cells, dtype=np.int64)

    for gene_id, indices in gene_to_tx_indices.items():
        if len(indices) < 2:
            continue

        if is_sparse:
            block = safe_toarray(
                X[:, indices], context=f"diversity block for gene {gene_id}")
        else:
            block = np.asarray(X[:, indices])

        # Only score cells where at least one isoform is used
        row_sums = block.sum(axis=1)
        active_mask = row_sums > 0

        if active_mask.sum() == 0:
            continue

        proportions = block[active_mask]

        if metric == 'shannon':
            # H = -sum(p * log2(p)), treating 0*log(0) = 0
            with np.errstate(divide='ignore', invalid='ignore'):
                log_p = np.log2(proportions, where=proportions > 0,
                                out=np.zeros_like(proportions))
            h = -np.sum(proportions * log_p, axis=1)
        else:
            # Simpson: D = 1 - sum(p^2)
            h = 1.0 - np.sum(proportions ** 2, axis=1)

        diversity_scores[active_mask] += h
        gene_counts[active_mask] += 1

    # Average across genes
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_diversity = np.where(
            gene_counts > 0, diversity_scores / gene_counts, 0.0)

    barcodes = adata_usage.obs_names.tolist()
    result_df = pd.DataFrame({
        'barcode': barcodes,
        'diversity_index': avg_diversity,
        'n_genes_multi_isoform': gene_counts,
    })

    logger.info(
        f"Diversity computed for {n_cells} cells; median "
        f"{metric}={np.median(avg_diversity):.4f}, "
        f"median multi-isoform genes={np.median(gene_counts):.0f}.")
    return result_df


def run_clustering(adata, method, resolution, n_neighbors, n_pcs,
                   random_state=0):
    """Run standard Scanpy gene-expression clustering pipeline.

    HVG selection, log-normalization, scaling, PCA, neighbors,
    Leiden/Louvain clustering, and UMAP.

    :returns: adata with 'cluster' in obs and UMAP in obsm.
    """
    import scanpy as sc

    logger = get_named_logger("GenClust")

    # Normalize
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # HVG selection
    logger.info("Selecting highly variable genes.")
    sc.pp.highly_variable_genes(
        adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    n_hvg = adata.var.highly_variable.sum()
    if n_hvg < 2:
        logger.warning(
            f"Only {n_hvg} highly variable genes found. "
            f"Using all {adata.shape[1]} genes instead.")
    else:
        adata.raw = adata
        adata = adata[:, adata.var.highly_variable].copy()

    # Scale — scanpy >= 1.9 handles sparse natively
    sc.pp.scale(adata, max_value=10)

    # PCA
    n_comps = min(n_pcs, adata.shape[0] - 1, adata.shape[1] - 1)
    if n_comps < 2:
        logger.warning(
            f"Too few features ({adata.shape[1]}) or cells "
            f"({adata.shape[0]}) for PCA. Assigning all to cluster 0.")
        adata.obs['cluster'] = '0'
        adata.obsm['X_umap'] = np.zeros((adata.shape[0], 2))
        return adata
    logger.info(f"Running PCA with {n_comps} components.")
    sc.tl.pca(adata, n_comps=n_comps, random_state=random_state)

    # Neighbors
    n_neighbors_safe = min(n_neighbors, adata.shape[0] - 1)
    logger.info(
        f"Building neighborhood graph: "
        f"n_neighbors={n_neighbors_safe}, n_pcs={n_comps}.")
    sc.pp.neighbors(
        adata, n_neighbors=n_neighbors_safe, n_pcs=n_comps,
        random_state=random_state)

    # Clustering
    logger.info(
        f"Clustering with {method} (resolution={resolution}).")
    if method == "leiden":
        sc.tl.leiden(
            adata, resolution=resolution, key_added='cluster',
            flavor='igraph', n_iterations=2, directed=False,
            random_state=random_state)
    else:
        sc.tl.louvain(
            adata, resolution=resolution, key_added='cluster',
            flavor='igraph', random_state=random_state)

    # UMAP
    logger.info("Computing UMAP embedding.")
    sc.tl.umap(adata, random_state=random_state)

    return adata


def run_isoform_clustering(adata_usage, method, resolution, n_neighbors,
                           n_pcs, random_state=0):
    """Cluster cells based on isoform usage proportions.

    Unlike gene-expression clustering, this skips log-normalization
    (data are already proportions) and HVG selection (all features are
    informative isoform ratios). Applies scaling, PCA, neighbors,
    clustering, and UMAP.

    :returns: adata_usage with 'cluster' in obs and UMAP in obsm.
    """
    import scanpy as sc

    logger = get_named_logger("IsoClust")

    if adata_usage.shape[1] == 0:
        logger.warning(
            "Empty isoform usage matrix; skipping isoform clustering.")
        adata_usage.obs['cluster'] = '0'
        return adata_usage

    # No log-normalization — already proportions
    # No HVG — all isoform ratios are informative

    # Scale — scanpy >= 1.9 handles sparse natively
    sc.pp.scale(adata_usage, max_value=10)

    # PCA
    n_comps = min(n_pcs, adata_usage.shape[0] - 1,
                  adata_usage.shape[1] - 1)
    if n_comps < 2:
        logger.warning(
            f"Too few features ({adata_usage.shape[1]}) or cells "
            f"({adata_usage.shape[0]}) for PCA. "
            f"Assigning all to cluster 0.")
        adata_usage.obs['cluster'] = '0'
        adata_usage.obsm['X_umap'] = np.zeros(
            (adata_usage.shape[0], 2))
        return adata_usage
    logger.info(f"Running PCA with {n_comps} components.")
    sc.tl.pca(adata_usage, n_comps=n_comps, random_state=random_state)

    # Neighbors
    n_neighbors_safe = min(n_neighbors, adata_usage.shape[0] - 1)
    logger.info(
        f"Building neighborhood graph: "
        f"n_neighbors={n_neighbors_safe}, n_pcs={n_comps}.")
    sc.pp.neighbors(
        adata_usage, n_neighbors=n_neighbors_safe, n_pcs=n_comps,
        random_state=random_state)

    # Clustering
    logger.info(
        f"Clustering with {method} (resolution={resolution}).")
    if method == "leiden":
        sc.tl.leiden(
            adata_usage, resolution=resolution, key_added='cluster',
            flavor='igraph', random_state=random_state)
    else:
        sc.tl.louvain(
            adata_usage, resolution=resolution, key_added='cluster',
            flavor='igraph', random_state=random_state)

    # UMAP
    logger.info("Computing UMAP embedding.")
    sc.tl.umap(adata_usage, random_state=random_state)

    return adata_usage


def compute_joint_embedding(adata_gene, adata_usage, n_neighbors, n_pcs,
                            method, resolution, isoform_weight=1.5,
                            normalize_pcs=True, random_state=0):
    """Compute a joint embedding via KNN graph fusion.

    Instead of concatenating PCA spaces (which suffers from the curse
    of dimensionality when one modality is noisy), this function
    builds separate KNN graphs in gene and isoform space and merges
    them as a weighted combination of connectivity matrices. This
    preserves cell-state structure from whichever modality defines it.

    Inspired by Seurat v4 WNN (Hao et al., Cell 2021), simplified
    for efficiency: the isoform_weight parameter directly controls
    the relative influence of isoform-level neighborhood structure
    in the merged graph.

    :param isoform_weight: Weight for isoform graph relative to gene
        graph. Default 1.5 gives 60%/40% isoform/gene split.
    :param normalize_pcs: If True, L2-normalize each PCA space
        before computing KNN graphs.
    :returns: AnnData with joint clusters and UMAP.
    """
    import anndata
    import scanpy as sc

    logger = get_named_logger("JointEmbed")

    # Find common cells
    common_cells = sorted(
        set(adata_gene.obs_names) & set(adata_usage.obs_names))
    logger.info(
        f"Joint embedding: {len(common_cells)} cells in common "
        f"(gene={adata_gene.shape[0]}, isoform={adata_usage.shape[0]}).")

    if len(common_cells) == 0:
        logger.warning("No common cells; cannot compute joint embedding.")
        empty = anndata.AnnData(
            obs=pd.DataFrame(index=[]),
            obsm={'X_umap': np.empty((0, 2))})
        empty.obs['cluster'] = pd.Categorical([])
        return empty

    # Subset to common cells
    gene_sub = adata_gene[common_cells].copy()
    usage_sub = adata_usage[common_cells].copy()

    has_iso_pca = ('X_pca' in usage_sub.obsm
                   and usage_sub.obsm['X_pca'].shape[1] > 0)

    if not has_iso_pca:
        logger.warning(
            "No PCA found for isoform data; using gene embedding only.")
        # Fall back to gene-only embedding
        adata_joint = anndata.AnnData(
            obs=gene_sub.obs[[]].copy(),
            obsm={'X_pca': gene_sub.obsm['X_pca'].copy()})
        n_comps = min(
            n_pcs, adata_joint.obsm['X_pca'].shape[1],
            len(common_cells) - 1)
        sc.pp.neighbors(
            adata_joint, n_neighbors=n_neighbors, n_pcs=n_comps,
            use_rep='X_pca', random_state=random_state)
        if method == "leiden":
            sc.tl.leiden(
                adata_joint, resolution=resolution, key_added='cluster',
                flavor='igraph', n_iterations=2, directed=False,
                random_state=random_state)
        else:
            sc.tl.louvain(
                adata_joint, resolution=resolution, key_added='cluster',
                flavor='igraph', random_state=random_state)
        sc.tl.umap(adata_joint, random_state=random_state)
        return adata_joint

    # --- Build separate KNN graphs ---

    # Gene-space KNN
    adata_gene_tmp = anndata.AnnData(
        obs=pd.DataFrame(index=common_cells),
        obsm={'X_pca': gene_sub.obsm['X_pca'].copy()})
    n_gene_comps = min(
        n_pcs, adata_gene_tmp.obsm['X_pca'].shape[1],
        len(common_cells) - 1)
    n_neighbors_safe = min(n_neighbors, len(common_cells) - 1)
    sc.pp.neighbors(
        adata_gene_tmp, n_neighbors=n_neighbors_safe,
        n_pcs=n_gene_comps, use_rep='X_pca')
    logger.info(
        f"Gene KNN graph: {n_neighbors_safe} neighbors, "
        f"{n_gene_comps} PCs.")

    # Isoform-space KNN
    adata_iso_tmp = anndata.AnnData(
        obs=pd.DataFrame(index=common_cells),
        obsm={'X_pca': usage_sub.obsm['X_pca'].copy()})
    n_iso_comps = min(
        n_pcs, adata_iso_tmp.obsm['X_pca'].shape[1],
        len(common_cells) - 1)
    sc.pp.neighbors(
        adata_iso_tmp, n_neighbors=n_neighbors_safe,
        n_pcs=n_iso_comps, use_rep='X_pca')
    logger.info(
        f"Isoform KNN graph: {n_neighbors_safe} neighbors, "
        f"{n_iso_comps} PCs.")

    # --- Merge graphs with weighted combination ---
    gene_w = 1.0 / (1.0 + isoform_weight)
    iso_w = isoform_weight / (1.0 + isoform_weight)
    logger.info(
        f"Graph fusion weights: gene={gene_w:.3f}, "
        f"isoform={iso_w:.3f} (isoform_weight={isoform_weight}).")

    # Merge connectivities (binary neighbor indicator)
    gene_conn = adata_gene_tmp.obsp['connectivities']
    iso_conn = adata_iso_tmp.obsp['connectivities']
    joint_conn = gene_w * gene_conn + iso_w * iso_conn

    # Merge distances (weighted average where both have edges,
    # single-modality distance otherwise)
    gene_dist = adata_gene_tmp.obsp['distances']
    iso_dist = adata_iso_tmp.obsp['distances']
    # Normalize distances to [0,1] range for fair merging
    gene_max = gene_dist.max() if gene_dist.nnz > 0 else 1.0
    iso_max = iso_dist.max() if iso_dist.nnz > 0 else 1.0
    if gene_max > 0:
        gene_dist_norm = gene_dist / gene_max
    else:
        gene_dist_norm = gene_dist
    if iso_max > 0:
        iso_dist_norm = iso_dist / iso_max
    else:
        iso_dist_norm = iso_dist
    joint_dist = gene_w * gene_dist_norm + iso_w * iso_dist_norm

    # Create joint AnnData with merged graph
    # Store concatenated PCA for UMAP initialization
    joint_pca = np.hstack([
        gene_sub.obsm['X_pca'], usage_sub.obsm['X_pca']])
    adata_joint = anndata.AnnData(
        obs=pd.DataFrame(index=common_cells),
        obsm={'X_pca': joint_pca})
    adata_joint.obsp['connectivities'] = joint_conn
    adata_joint.obsp['distances'] = joint_dist
    adata_joint.uns['neighbors'] = {
        'connectivities_key': 'connectivities',
        'distances_key': 'distances',
        'params': {
            'n_neighbors': n_neighbors_safe,
            'method': 'graph_fusion',
        },
    }

    # Clustering on the merged graph
    logger.info(
        f"Joint clustering with {method} (resolution={resolution}) "
        f"on fused graph.")
    if method == "leiden":
        sc.tl.leiden(
            adata_joint, resolution=resolution, key_added='cluster',
            flavor='igraph', n_iterations=2, directed=False,
            random_state=random_state)
    else:
        sc.tl.louvain(
            adata_joint, resolution=resolution, key_added='cluster',
            flavor='igraph', random_state=random_state)

    # UMAP on the merged graph
    logger.info("Computing joint UMAP embedding on fused graph.")
    sc.tl.umap(adata_joint, random_state=random_state)

    return adata_joint


def compare_clusterings(gene_clusters, isoform_clusters):
    """Compare gene-level and isoform-level clusterings.

    Computes Adjusted Rand Index (ARI), Normalized Mutual Information
    (NMI), a contingency table, and identifies isoform-specific
    clusters that span multiple gene clusters.

    :param gene_clusters: Series with gene cluster labels, indexed by
        barcode.
    :param isoform_clusters: Series with isoform cluster labels,
        indexed by barcode.
    :returns: dict with comparison statistics.
    """
    from sklearn.metrics import (
        adjusted_rand_score, normalized_mutual_info_score)

    logger = get_named_logger("Compare")

    # Align to common barcodes
    common = sorted(set(gene_clusters.index) & set(isoform_clusters.index))
    if len(common) == 0:
        logger.warning("No common barcodes for comparison.")
        return {
            'ari': None, 'nmi': None,
            'n_common_cells': 0,
            'contingency': {},
            'isoform_specific_clusters': [],
        }

    g = gene_clusters.loc[common]
    iso = isoform_clusters.loc[common]

    ari = float(adjusted_rand_score(g, iso))
    nmi = float(normalized_mutual_info_score(g, iso))
    logger.info(f"ARI={ari:.4f}, NMI={nmi:.4f} over {len(common)} cells.")

    # Contingency table
    contingency = pd.crosstab(
        g.rename('gene_cluster'),
        iso.rename('isoform_cluster'))
    contingency_dict = {
        str(gc): {str(ic): int(v) for ic, v in row.items()}
        for gc, row in contingency.iterrows()}

    # Identify isoform-specific clusters: isoform clusters that span
    # multiple gene clusters (no single gene cluster holds > 60% of
    # the isoform cluster's cells)
    isoform_specific = []
    for iso_cl in contingency.columns:
        col = contingency[iso_cl]
        total = col.sum()
        if total == 0:
            continue
        max_frac = col.max() / total
        if max_frac <= 0.6:
            top_gene_clusters = col[col > 0].index.tolist()
            isoform_specific.append({
                'isoform_cluster': str(iso_cl),
                'n_cells': int(total),
                'max_gene_cluster_fraction': float(max_frac),
                'gene_clusters_spanned': [str(x) for x in top_gene_clusters],
            })
    logger.info(
        f"Found {len(isoform_specific)} isoform-specific clusters "
        f"(spanning multiple gene clusters).")

    return {
        'ari': ari,
        'nmi': nmi,
        'n_common_cells': len(common),
        'contingency': contingency_dict,
        'isoform_specific_clusters': isoform_specific,
    }


def main(args):
    """Run dual-layer clustering pipeline."""
    logger = get_named_logger("DualClust")
    logger.info("Starting dual-layer clustering.")

    # Load gene expression matrix
    logger.info(f"Loading gene matrix from {args.gene_matrix_dir}.")
    adata_gene = load_mex_to_anndata(args.gene_matrix_dir)
    logger.info(
        f"Gene matrix: {adata_gene.shape[0]} cells x "
        f"{adata_gene.shape[1]} genes.")

    # Load transcript expression matrix
    logger.info(
        f"Loading transcript matrix from {args.transcript_matrix_dir}.")
    adata_transcript = load_mex_to_anndata(args.transcript_matrix_dir)
    logger.info(
        f"Transcript matrix: {adata_transcript.shape[0]} cells x "
        f"{adata_transcript.shape[1]} transcripts.")

    # Load gene-transcript map
    gene_map = load_gene_transcript_map(args.gene_transcript_map)

    # Compute isoform usage proportions
    logger.info("Computing isoform usage proportions.")
    adata_usage = compute_isoform_usage(
        adata_transcript, gene_map,
        min_isoforms=args.min_isoforms_per_gene)

    # Compute diversity index
    logger.info("Computing isoform diversity index.")
    diversity_df = compute_diversity_index(
        adata_usage, gene_map, metric=args.diversity_metric)
    diversity_df.to_csv(args.output_diversity, sep='\t', index=False)
    logger.info(f"Diversity index written to {args.output_diversity}.")

    # Run gene-level clustering
    logger.info("Running gene-level clustering.")
    adata_gene_clust = run_clustering(
        adata_gene, method=args.cluster_method,
        resolution=args.resolution,
        n_neighbors=args.n_neighbors, n_pcs=args.n_pcs,
        random_state=args.random_state)
    gene_clusters_df = pd.DataFrame({
        'barcode': adata_gene_clust.obs_names,
        'cluster': adata_gene_clust.obs['cluster'].values,
    })
    gene_clusters_df.to_csv(
        args.output_gene_clusters, sep='\t', index=False)
    logger.info(
        f"Gene clustering: {adata_gene_clust.obs['cluster'].nunique()} "
        f"clusters across {adata_gene_clust.shape[0]} cells.")

    # Run isoform-usage clustering
    logger.info("Running isoform-usage clustering.")
    adata_iso_clust = run_isoform_clustering(
        adata_usage, method=args.cluster_method,
        resolution=args.isoform_resolution,
        n_neighbors=args.n_neighbors, n_pcs=args.n_pcs,
        random_state=args.random_state)
    iso_clusters_df = pd.DataFrame({
        'barcode': adata_iso_clust.obs_names,
        'cluster': adata_iso_clust.obs['cluster'].values,
    })
    iso_clusters_df.to_csv(
        args.output_isoform_clusters, sep='\t', index=False)
    logger.info(
        f"Isoform clustering: {adata_iso_clust.obs['cluster'].nunique()} "
        f"clusters across {adata_iso_clust.shape[0]} cells.")

    # Compute joint embedding
    logger.info("Computing joint gene+isoform embedding.")
    isoform_weight = getattr(args, 'isoform_weight', 1.5)
    normalize_pcs = getattr(args, 'normalize_pcs', True)
    adata_joint = compute_joint_embedding(
        adata_gene_clust, adata_iso_clust,
        n_neighbors=args.n_neighbors, n_pcs=args.n_pcs,
        method=args.cluster_method, resolution=args.resolution,
        isoform_weight=isoform_weight, normalize_pcs=normalize_pcs,
        random_state=args.random_state)

    if adata_joint.shape[0] > 0:
        joint_clusters_df = pd.DataFrame({
            'barcode': adata_joint.obs_names,
            'cluster': adata_joint.obs['cluster'].values,
        })
        joint_clusters_df.to_csv(
            args.output_joint_clusters, sep='\t', index=False)

        umap_coords = adata_joint.obsm['X_umap']
        umap_df = pd.DataFrame(
            umap_coords, columns=['D1', 'D2'],
            index=adata_joint.obs_names)
        umap_df['cluster'] = adata_joint.obs['cluster'].values
        umap_df.to_csv(
            args.output_joint_umap, sep='\t', index_label='CB')
        logger.info(
            f"Joint clustering: {adata_joint.obs['cluster'].nunique()} "
            f"clusters across {adata_joint.shape[0]} cells.")
    else:
        logger.warning("Joint embedding produced no cells; skipping output.")

    # Compare gene vs isoform clusterings
    logger.info("Comparing gene and isoform clusterings.")
    gene_cl = pd.Series(
        adata_gene_clust.obs['cluster'].values,
        index=adata_gene_clust.obs_names, name='gene_cluster')
    iso_cl = pd.Series(
        adata_iso_clust.obs['cluster'].values,
        index=adata_iso_clust.obs_names, name='isoform_cluster')
    comparison = compare_clusterings(gene_cl, iso_cl)

    with open(args.output_comparison, 'w') as fh:
        json.dump(comparison, fh, indent=2)
    logger.info(f"Cluster comparison written to {args.output_comparison}.")

    logger.info("Dual-layer clustering complete.")
