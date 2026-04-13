"""Automated cell type annotation using marker gene databases.

Assigns cell type labels to clusters by comparing cluster expression
profiles against known marker gene sets. Supports two methods:
marker gene overlap scoring and correlation-based annotation.

Includes built-in marker gene databases for human and mouse, with
support for custom marker gene TSV files.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats

from ._logging import get_named_logger, wf_parser


# ---------------------------------------------------------------------------
# Default marker gene databases — curated from CellMarker 2.0, PanglaoDB,
# the Human Protein Atlas, and canonical literature (Zheng 2017, Tabula
# Sapiens 2022, Tabula Muris 2018).  Each type has 8-25 markers spanning
# surface proteins, transcription factors, and secreted factors to
# maximise detection across platforms and normalisation methods.
# ---------------------------------------------------------------------------

_HUMAN_MARKERS = {
    'CD4 T cells': [
        'CD3D', 'CD3E', 'CD3G', 'CD4', 'IL7R', 'LEF1', 'TCF7', 'MAL',
        'LTB', 'TRAC', 'LDHB', 'CD27', 'NOSIP', 'PIK3IP1',
    ],
    'CD8 T cells': [
        'CD3D', 'CD3E', 'CD8A', 'CD8B', 'GZMK', 'GZMA', 'CCL5', 'NKG7',
        'CST7', 'GZMH', 'CTSW', 'KLRG1', 'LYAR', 'TRGC2',
    ],
    'Regulatory T cells': [
        'FOXP3', 'IL2RA', 'CTLA4', 'TIGIT', 'IKZF2', 'TNFRSF18',
        'TNFRSF4', 'BATF', 'IL32', 'CD3D',
    ],
    'Naive B cells': [
        'MS4A1', 'CD79A', 'CD79B', 'CD19', 'BANK1', 'IGHD', 'IGHM',
        'TCL1A', 'FCER2', 'CD22', 'PAX5', 'VPREB3',
    ],
    'Memory B cells': [
        'MS4A1', 'CD79A', 'CD79B', 'CD27', 'IGHG1', 'IGHG2', 'AIM2',
        'TNFRSF13B', 'CD19',
    ],
    'Plasma cells': [
        'MZB1', 'SDC1', 'JCHAIN', 'IGHG1', 'XBP1', 'PRDM1', 'IRF4',
        'CD38', 'TNFRSF17',
    ],
    'NK cells': [
        'GNLY', 'NKG7', 'KLRD1', 'KLRB1', 'KLRF1', 'PRF1', 'GZMB',
        'NCAM1', 'FGFBP2', 'SPON2', 'CLIC3', 'FCGR3A', 'TYROBP',
        'CD247', 'TRDC',
    ],
    'CD14+ Monocytes': [
        'CD14', 'LYZ', 'S100A9', 'S100A8', 'VCAN', 'FCN1', 'S100A12',
        'CTSS', 'CD68', 'MNDA', 'TYROBP', 'CST3', 'LGALS2', 'MS4A6A',
    ],
    'FCGR3A+ Monocytes': [
        'FCGR3A', 'MS4A7', 'LST1', 'IFITM3', 'AIF1', 'COTL1', 'LYPD2',
        'LILRA3', 'FCER1G', 'VMO1', 'SERPINA1', 'CDKN1C',
    ],
    'Dendritic cells': [
        'FCER1A', 'CST3', 'HLA-DQA1', 'HLA-DPB1', 'HLA-DRA', 'HLA-DPA1',
        'CLEC10A', 'CD1C', 'ITGAX', 'IRF8', 'IRF4', 'LILRA4', 'IL3RA',
        'JCHAIN',
    ],
    'Plasmacytoid DCs': [
        'LILRA4', 'IL3RA', 'CLEC4C', 'IRF7', 'TCF4', 'PLD4', 'SERPINF1',
        'ITM2C', 'GZMB', 'JCHAIN',
    ],
    'Platelets': [
        'PPBP', 'PF4', 'GP9', 'ITGA2B', 'TUBB1', 'SPARC', 'GNG11',
        'CLU', 'HIST1H2AC', 'ACRBP', 'TMEM40', 'PTCRA',
    ],
    'Erythrocytes': [
        'HBB', 'HBA1', 'HBA2', 'HBD', 'ALAS2', 'SLC4A1', 'CA1',
        'GYPA', 'AHSP', 'EPB41', 'SNCA', 'ANK1',
    ],
    'Macrophages': [
        'CD68', 'MARCO', 'MRC1', 'MSR1', 'CD163', 'MERTK', 'C1QA',
        'C1QB', 'C1QC', 'APOE', 'LGMN', 'FCGR1A', 'SIGLEC1',
    ],
    'Neutrophils': [
        'S100A8', 'S100A9', 'S100A12', 'FCGR3B', 'CSF3R', 'CXCR2',
        'NAMPT', 'MMP9', 'FPR1', 'SELL', 'CEACAM8', 'SLC25A37',
    ],
    'Mast cells': [
        'TPSB2', 'TPSAB1', 'CPA3', 'KIT', 'MS4A2', 'HPGDS', 'GATA2',
        'HDC', 'CLU',
    ],
    'Fibroblasts': [
        'COL1A1', 'COL1A2', 'COL3A1', 'DCN', 'LUM', 'PDGFRA', 'THY1',
        'FAP', 'VIM', 'FN1', 'S100A4', 'ACTA2',
    ],
    'Epithelial cells': [
        'EPCAM', 'KRT18', 'KRT19', 'KRT8', 'CDH1', 'CLDN4', 'MUC1',
        'TFF3', 'PIGR', 'WFDC2', 'KRT7',
    ],
    'Endothelial cells': [
        'PECAM1', 'VWF', 'CDH5', 'CLDN5', 'ESAM', 'EMCN', 'ERG',
        'FLT1', 'KDR', 'PLVAP', 'ACKR1', 'SELE',
    ],
    'Smooth muscle cells': [
        'ACTA2', 'MYH11', 'TAGLN', 'CNN1', 'MYLK', 'DES', 'ACTG2',
        'MYL9', 'TPM2',
    ],
}

_MOUSE_MARKERS = {
    'CD4 T cells': [
        'Cd3d', 'Cd3e', 'Cd3g', 'Cd4', 'Il7r', 'Lef1', 'Tcf7', 'Mal',
        'Ltb', 'Trac', 'Ldhb', 'Cd27',
    ],
    'CD8 T cells': [
        'Cd3d', 'Cd3e', 'Cd8a', 'Cd8b1', 'Gzmk', 'Gzma', 'Ccl5',
        'Nkg7', 'Cst7', 'Klrg1',
    ],
    'B cells': [
        'Ms4a1', 'Cd79a', 'Cd79b', 'Cd19', 'Bank1', 'Ighd', 'Ighm',
        'Pax5', 'Cd22', 'Ebf1', 'Vpreb3',
    ],
    'Plasma cells': [
        'Mzb1', 'Sdc1', 'Jchain', 'Xbp1', 'Prdm1', 'Irf4', 'Cd38',
    ],
    'NK cells': [
        'Gzma', 'Nkg7', 'Klrb1c', 'Klrd1', 'Prf1', 'Gzmb', 'Ncam1',
        'Klrk1', 'Tyrobp', 'Cd247',
    ],
    'Monocytes': [
        'Cd14', 'Lyz2', 'S100a9', 'S100a8', 'Vcan', 'Fcn1', 'Cst3',
        'Ctss', 'Lgals3', 'Csf1r', 'Ms4a6c',
    ],
    'Dendritic cells': [
        'Fcer1a', 'Cst3', 'H2-Aa', 'H2-Ab1', 'H2-Eb1', 'Cd74',
        'Itgax', 'Irf8', 'Irf4', 'Siglech',
    ],
    'Macrophages': [
        'Cd68', 'Adgre1', 'Mrc1', 'Marco', 'Cd163', 'C1qa', 'C1qb',
        'C1qc', 'Apoe', 'Fcgr1',
    ],
    'Neutrophils': [
        'S100a8', 'S100a9', 'Cxcr2', 'Csf3r', 'Mmp9', 'Retnlg',
        'Sell', 'Camp',
    ],
    'Platelets': [
        'Ppbp', 'Pf4', 'Gp9', 'Itga2b', 'Tubb1', 'Gng11',
    ],
    'Erythrocytes': [
        'Hbb-bs', 'Hba-a1', 'Hba-a2', 'Alas2', 'Car1', 'Gypa',
        'Snca', 'Ank1',
    ],
    'Fibroblasts': [
        'Col1a1', 'Col1a2', 'Col3a1', 'Dcn', 'Lum', 'Pdgfra',
        'Fn1', 'Vim', 'S100a4',
    ],
    'Epithelial cells': [
        'Epcam', 'Krt18', 'Krt19', 'Krt8', 'Cdh1', 'Cldn4', 'Muc1',
    ],
    'Endothelial cells': [
        'Pecam1', 'Vwf', 'Cdh5', 'Cldn5', 'Emcn', 'Erg', 'Flt1',
        'Kdr', 'Plvap',
    ],
}


def argparser():
    """Create argument parser."""
    parser = wf_parser("cell_type_annotation")

    parser.add_argument(
        "gene_matrix_dir", type=Path,
        help="Path to processed gene MEX matrix directory.")
    parser.add_argument(
        "--clusters", type=Path, required=True,
        help="TSV with columns barcode, cluster.")
    parser.add_argument(
        "--marker_genes_db", type=Path, default=None,
        help="Custom marker genes TSV with columns cell_type, gene.")
    parser.add_argument(
        "--output_annotations", type=Path,
        default="cell_type_annotations.tsv",
        help="Output TSV with per-cell type annotations.")
    parser.add_argument(
        "--output_cluster_types", type=Path,
        default="cluster_cell_types.tsv",
        help="Output TSV with cell type assignment per cluster.")
    parser.add_argument(
        "--output_summary", type=Path,
        default="cell_type_summary.json",
        help="Output JSON with annotation summary statistics.")
    parser.add_argument(
        "--method", default="marker_overlap",
        choices=["marker_overlap", "correlation"],
        help="Annotation method.")
    parser.add_argument(
        "--min_marker_genes", type=int, default=2,
        help="Minimum marker genes detected to assign a cell type.")
    parser.add_argument(
        "--cluster_column", type=str, default="cluster",
        help="Column name for cluster labels in clusters TSV.")
    parser.add_argument(
        "--species", default="human",
        choices=["human", "mouse"],
        help="Species for default marker gene database.")

    return parser


def get_default_markers(species):
    """Return default marker gene database for the given species.

    :param species: 'human' or 'mouse'.
    :returns: dict mapping cell_type -> list of marker gene symbols.
    """
    if species == 'mouse':
        return {k: list(v) for k, v in _MOUSE_MARKERS.items()}
    return {k: list(v) for k, v in _HUMAN_MARKERS.items()}


def load_custom_markers(path):
    """Load custom marker genes from a TSV file.

    Expected columns: cell_type, gene. Additional columns are ignored.

    :param path: Path to TSV file.
    :returns: dict mapping cell_type -> list of gene symbols.
    """
    logger = get_named_logger("Markers")

    df = pd.read_csv(path, sep='\t')
    if 'cell_type' not in df.columns or 'gene' not in df.columns:
        raise ValueError(
            "Custom marker TSV must have 'cell_type' and 'gene' columns. "
            f"Found columns: {list(df.columns)}")

    markers = {}
    for ct, grp in df.groupby('cell_type'):
        markers[ct] = sorted(grp['gene'].unique().tolist())

    logger.info(
        f"Loaded {len(markers)} cell types with "
        f"{sum(len(v) for v in markers.values())} total marker genes "
        f"from {path}.")
    return markers


def _get_cluster_profiles(adata, cluster_labels):
    """Compute mean expression profile per cluster.

    :param adata: AnnData with gene expression.
    :param cluster_labels: Series mapping barcode -> cluster.
    :returns: DataFrame (clusters x genes) of mean expression values.
    """
    common = sorted(set(adata.obs_names) & set(cluster_labels.index))
    adata_sub = adata[common].copy()
    clusters = cluster_labels.loc[common]
    unique_clusters = sorted(clusters.unique())

    X = adata_sub.X
    profiles = {}

    for cluster in unique_clusters:
        mask = np.array(clusters == cluster)
        if scipy.sparse.issparse(X):
            cluster_mean = np.asarray(X[mask].mean(axis=0)).ravel()
        else:
            cluster_mean = np.mean(X[mask], axis=0).ravel()
        profiles[cluster] = cluster_mean

    profile_df = pd.DataFrame(
        profiles, index=adata_sub.var_names).T
    return profile_df


def annotate_clusters_by_overlap(
        adata, cluster_labels, marker_db, min_markers=3):
    """Annotate clusters using marker gene overlap scoring.

    For each cluster, scores each candidate cell type using a composite
    metric that accounts for:
      1. Fraction of marker genes detected (specificity)
      2. Mean log-normalised expression of detected markers
      3. Enrichment relative to other clusters (z-score of marker
         expression vs. global mean)

    The composite score is: fraction_detected × mean_expression × (1 + z_score).
    The highest-scoring cell type is assigned if at least min_markers
    genes are detected.

    :param adata: AnnData with gene expression.
    :param cluster_labels: Series mapping barcode -> cluster.
    :param marker_db: dict mapping cell_type -> list of gene symbols.
    :param min_markers: minimum detected markers to assign a type.
    :returns: DataFrame with columns cluster, cell_type, score,
        n_markers_detected, markers_found.
    """
    logger = get_named_logger("Overlap")

    profile_df = _get_cluster_profiles(adata, cluster_labels)
    available_genes = set(profile_df.columns)
    unique_clusters = sorted(profile_df.index)

    # Precompute global mean expression per gene (across all clusters)
    global_mean = profile_df.mean(axis=0)
    global_std = profile_df.std(axis=0).replace(0, 1)  # avoid div by 0

    results = []

    for cluster in unique_clusters:
        cluster_profile = profile_df.loc[cluster]
        best_type = "Unknown"
        best_score = -1.0
        best_n_detected = 0
        best_markers = []

        for cell_type, markers in marker_db.items():
            found = [g for g in markers if g in available_genes]
            if len(found) == 0:
                continue

            detected = [
                g for g in found if cluster_profile[g] > 0]
            n_detected = len(detected)

            if n_detected == 0:
                continue

            # Fraction of available markers that are detected
            frac_detected = n_detected / len(found)

            # Mean expression of detected markers in this cluster
            mean_expr = float(cluster_profile[detected].mean())

            # Enrichment: z-score of marker expression vs global mean
            z_scores = [
                (cluster_profile[g] - global_mean[g]) / global_std[g]
                for g in detected
            ]
            mean_z = float(np.mean(z_scores)) if z_scores else 0.0

            # Composite score: specificity × expression × enrichment
            score = frac_detected * mean_expr * (1.0 + max(mean_z, 0.0))

            if score > best_score:
                best_score = score
                best_type = cell_type
                best_n_detected = n_detected
                best_markers = detected

        if best_n_detected < min_markers:
            best_type = "Unknown"

        results.append({
            'cluster': cluster,
            'cell_type': best_type,
            'score': round(best_score, 4),
            'n_markers_detected': best_n_detected,
            'markers_found': ','.join(best_markers),
        })

    result_df = pd.DataFrame(results)
    n_annotated = int((result_df['cell_type'] != 'Unknown').sum())
    logger.info(
        f"Overlap annotation: {n_annotated}/{len(unique_clusters)} "
        f"clusters annotated.")
    return result_df


def annotate_clusters_by_correlation(
        adata, cluster_labels, marker_db):
    """Annotate clusters using Pearson correlation against reference profiles.

    Constructs binary reference profiles from the marker gene database
    and correlates each cluster's mean expression profile against them.
    The cell type with the highest positive correlation is assigned.

    :param adata: AnnData with gene expression.
    :param cluster_labels: Series mapping barcode -> cluster.
    :param marker_db: dict mapping cell_type -> list of gene symbols.
    :returns: DataFrame with columns cluster, cell_type, correlation,
        pvalue.
    """
    logger = get_named_logger("Corr")

    profile_df = _get_cluster_profiles(adata, cluster_labels)
    available_genes = list(profile_df.columns)
    gene_set = set(available_genes)
    unique_clusters = sorted(profile_df.index)

    # Build binary reference profiles (genes x cell_types)
    ref_profiles = {}
    for cell_type, markers in marker_db.items():
        ref_vec = np.zeros(len(available_genes))
        for gene in markers:
            if gene in gene_set:
                idx = available_genes.index(gene)
                ref_vec[idx] = 1.0
        if ref_vec.sum() > 0:
            ref_profiles[cell_type] = ref_vec

    if not ref_profiles:
        logger.warning(
            "No marker genes found in expression matrix. "
            "All clusters will be labelled Unknown.")
        return pd.DataFrame({
            'cluster': unique_clusters,
            'cell_type': ['Unknown'] * len(unique_clusters),
            'correlation': [0.0] * len(unique_clusters),
            'pvalue': [1.0] * len(unique_clusters),
        })

    results = []

    for cluster in unique_clusters:
        cluster_vec = profile_df.loc[cluster].values.astype(float)
        best_type = "Unknown"
        best_corr = -1.0
        best_pval = 1.0

        for cell_type, ref_vec in ref_profiles.items():
            # Pearson correlation
            if np.std(cluster_vec) == 0 or np.std(ref_vec) == 0:
                continue
            r, pval = scipy.stats.pearsonr(cluster_vec, ref_vec)

            if r > best_corr:
                best_corr = r
                best_type = cell_type
                best_pval = pval

        if best_corr <= 0:
            best_type = "Unknown"

        results.append({
            'cluster': cluster,
            'cell_type': best_type,
            'correlation': round(best_corr, 4),
            'pvalue': best_pval,
        })

    result_df = pd.DataFrame(results)
    n_annotated = int((result_df['cell_type'] != 'Unknown').sum())
    logger.info(
        f"Correlation annotation: {n_annotated}/{len(unique_clusters)} "
        f"clusters annotated.")
    return result_df


def _build_per_cell_annotations(cluster_labels, cluster_types_df):
    """Map cluster-level annotations to individual cells.

    :param cluster_labels: Series mapping barcode -> cluster.
    :param cluster_types_df: DataFrame with cluster, cell_type columns.
    :returns: DataFrame with barcode, cluster, cell_type columns.
    """
    type_map = dict(zip(
        cluster_types_df['cluster'].astype(str),
        cluster_types_df['cell_type']))

    records = []
    for barcode, cluster in cluster_labels.items():
        records.append({
            'barcode': barcode,
            'cluster': cluster,
            'cell_type': type_map.get(str(cluster), 'Unknown'),
        })
    return pd.DataFrame(records)


def main(args):
    """Run cell type annotation pipeline."""
    import scanpy as sc

    logger = get_named_logger("CellType")
    logger.info("Starting cell type annotation.")

    # Load gene expression matrix
    logger.info(f"Loading gene matrix from {args.gene_matrix_dir}.")
    adata = sc.read_10x_mtx(
        str(args.gene_matrix_dir), var_names='gene_symbols')
    adata.var_names_make_unique()
    logger.info(
        f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes.")

    # Normalize for annotation
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Load cluster assignments
    clusters_df = pd.read_csv(args.clusters, sep='\t')
    barcode_col = clusters_df.columns[0]
    cluster_col = args.cluster_column
    cluster_labels = pd.Series(
        clusters_df[cluster_col].astype(str).values,
        index=clusters_df[barcode_col].values)
    logger.info(
        f"Loaded {len(cluster_labels)} cells in "
        f"{cluster_labels.nunique()} clusters.")

    # Load or build marker gene database
    if args.marker_genes_db is not None:
        marker_db = load_custom_markers(args.marker_genes_db)
    else:
        marker_db = get_default_markers(args.species)
        logger.info(
            f"Using default {args.species} marker database "
            f"({len(marker_db)} cell types).")

    # Annotate clusters
    if args.method == 'marker_overlap':
        cluster_types_df = annotate_clusters_by_overlap(
            adata, cluster_labels, marker_db,
            min_markers=args.min_marker_genes)
    else:
        cluster_types_df = annotate_clusters_by_correlation(
            adata, cluster_labels, marker_db)

    # Write cluster-level annotations
    cluster_types_df.to_csv(
        args.output_cluster_types, sep='\t', index=False)
    logger.info(
        f"Cluster annotations written to {args.output_cluster_types}.")

    # Build and write per-cell annotations
    cell_annotations = _build_per_cell_annotations(
        cluster_labels, cluster_types_df)
    cell_annotations.to_csv(
        args.output_annotations, sep='\t', index=False)
    logger.info(
        f"Per-cell annotations written to {args.output_annotations}.")

    # Summary statistics
    type_counts = cell_annotations['cell_type'].value_counts().to_dict()
    n_annotated_clusters = int(
        (cluster_types_df['cell_type'] != 'Unknown').sum())
    n_total_clusters = len(cluster_types_df)

    summary = {
        'method': args.method,
        'species': args.species,
        'custom_markers': args.marker_genes_db is not None,
        'n_marker_cell_types': len(marker_db),
        'n_clusters': n_total_clusters,
        'n_annotated_clusters': n_annotated_clusters,
        'n_unknown_clusters': n_total_clusters - n_annotated_clusters,
        'n_cells': len(cell_annotations),
        'cell_type_counts': type_counts,
        'cluster_annotations': cluster_types_df.to_dict('records'),
    }
    with open(args.output_summary, 'w') as fh:
        json.dump(summary, fh, indent=2)

    logger.info(
        f"Cell type annotation complete: "
        f"{n_annotated_clusters}/{n_total_clusters} clusters annotated.")
