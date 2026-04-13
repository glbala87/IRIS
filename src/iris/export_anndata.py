"""Export unified AnnData h5ad for the isosceles pipeline.

Aggregates all analysis outputs (gene/transcript matrices, clustering,
UMAP embeddings, cell type annotations, isoform diversity, DTU results,
novel isoform catalogs, pseudotime, and ASE results) into a single
AnnData h5ad file for downstream exploration and visualization.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse

from ._logging import get_named_logger, wf_parser


def argparser():
    """Create argument parser."""
    parser = wf_parser("export_anndata")

    parser.add_argument(
        "gene_matrix_dir", type=Path,
        help="Path to processed gene MEX matrix directory.")
    parser.add_argument(
        "--transcript_matrix_dir", type=Path, default=None,
        help="Path to processed transcript MEX matrix directory.")
    parser.add_argument(
        "--gene_clusters", type=Path, default=None,
        help="TSV with gene-level cluster assignments.")
    parser.add_argument(
        "--isoform_clusters", type=Path, default=None,
        help="TSV with isoform-level cluster assignments.")
    parser.add_argument(
        "--joint_clusters", type=Path, default=None,
        help="TSV with joint gene+isoform cluster assignments.")
    parser.add_argument(
        "--joint_umap", type=Path, default=None,
        help="TSV with joint UMAP coordinates.")
    parser.add_argument(
        "--cell_type_annotations", type=Path, default=None,
        help="TSV with cell type annotations per cell.")
    parser.add_argument(
        "--isoform_diversity", type=Path, default=None,
        help="TSV with isoform diversity metrics per cell.")
    parser.add_argument(
        "--dtu_results", type=Path, default=None,
        help="TSV with differential transcript usage results.")
    parser.add_argument(
        "--switching_results", type=Path, default=None,
        help="TSV with isoform switching results.")
    parser.add_argument(
        "--novel_catalog", type=Path, default=None,
        help="TSV with novel isoform catalog.")
    parser.add_argument(
        "--novel_enrichment", type=Path, default=None,
        help="TSV with novel isoform cluster enrichment.")
    parser.add_argument(
        "--cluster_comparison", type=Path, default=None,
        help="TSV with cluster comparison results.")
    parser.add_argument(
        "--pseudotime", type=Path, default=None,
        help="TSV with pseudotime values per cell.")
    parser.add_argument(
        "--ase_results", type=Path, default=None,
        help="TSV with allele-specific expression results.")
    parser.add_argument(
        "--output", type=Path, default="iris.h5ad",
        help="Output h5ad file path.")

    return parser


def load_and_merge_clusters(adata, files_dict):
    """Load cluster TSVs and add as columns in adata.obs.

    Each file is expected to have a barcode column (first column) and a
    cluster column. The column is added to adata.obs with the key from
    files_dict.

    :param adata: AnnData object.
    :param files_dict: dict mapping obs column name -> Path to TSV.
    :returns: adata with added obs columns.
    """
    logger = get_named_logger("Clusters")

    for col_name, tsv_path in files_dict.items():
        if tsv_path is None or not Path(tsv_path).exists():
            continue

        df = pd.read_csv(tsv_path, sep='\t')
        barcode_col = df.columns[0]
        value_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        mapping = dict(zip(
            df[barcode_col].astype(str),
            df[value_col].astype(str)))

        adata.obs[col_name] = adata.obs_names.map(
            lambda x: mapping.get(x, 'NA'))
        n_mapped = int((adata.obs[col_name] != 'NA').sum())
        logger.info(
            f"Added '{col_name}': {n_mapped}/{adata.shape[0]} cells "
            f"mapped from {tsv_path}.")

    return adata


def add_umap_embedding(adata, umap_file, key='X_umap'):
    """Add UMAP coordinates to adata.obsm.

    The TSV should have a barcode index and columns D1, D2 (or UMAP1, UMAP2).

    :param adata: AnnData object.
    :param umap_file: Path to UMAP TSV.
    :param key: key in adata.obsm for the embedding.
    :returns: adata with added obsm entry.
    """
    logger = get_named_logger("UMAP")

    if umap_file is None or not Path(umap_file).exists():
        return adata

    df = pd.read_csv(umap_file, sep='\t', index_col=0)

    # Standardize column names
    if 'D1' in df.columns and 'D2' in df.columns:
        coord_cols = ['D1', 'D2']
    elif 'UMAP1' in df.columns and 'UMAP2' in df.columns:
        coord_cols = ['UMAP1', 'UMAP2']
    else:
        coord_cols = df.columns[:2].tolist()

    common = sorted(set(adata.obs_names) & set(df.index))
    if len(common) == 0:
        logger.warning(f"No overlapping barcodes in {umap_file}.")
        return adata

    coords = df.loc[common, coord_cols].values.astype(np.float32)

    # Initialize with zeros, fill matched cells
    embedding = np.zeros((adata.shape[0], 2), dtype=np.float32)
    obs_idx = {bc: i for i, bc in enumerate(adata.obs_names)}
    for j, bc in enumerate(common):
        embedding[obs_idx[bc]] = coords[j]

    adata.obsm[key] = embedding
    logger.info(
        f"Added UMAP embedding '{key}': {len(common)} cells "
        f"from {umap_file}.")
    return adata


def add_cell_metadata(adata, tsv_file, columns=None):
    """Add per-cell metadata from a TSV file to adata.obs.

    :param adata: AnnData object.
    :param tsv_file: Path to TSV with barcode as first column.
    :param columns: list of column names to add. If None, adds all.
    :returns: adata with added obs columns.
    """
    logger = get_named_logger("Metadata")

    if tsv_file is None or not Path(tsv_file).exists():
        return adata

    df = pd.read_csv(tsv_file, sep='\t')
    barcode_col = df.columns[0]
    df = df.set_index(barcode_col)

    if columns is not None:
        cols_present = [c for c in columns if c in df.columns]
        df = df[cols_present]

    common = sorted(set(adata.obs_names) & set(df.index))
    if len(common) == 0:
        logger.warning(f"No overlapping barcodes in {tsv_file}.")
        return adata

    for col in df.columns:
        mapping = df[col].to_dict()
        adata.obs[col] = adata.obs_names.map(
            lambda x, m=mapping: m.get(x, np.nan))

    logger.info(
        f"Added {len(df.columns)} metadata columns from {tsv_file} "
        f"({len(common)} cells matched).")
    return adata


def add_uns_dataframe(adata, key, tsv_file):
    """Store a DataFrame in adata.uns for non-per-cell results.

    Supports both TSV and JSON files. JSON files are stored as
    serialized strings since h5ad has limited support for nested dicts.

    :param adata: AnnData object.
    :param key: key in adata.uns.
    :param tsv_file: Path to TSV or JSON file.
    :returns: adata with added uns entry.
    """
    logger = get_named_logger("UnsData")

    if tsv_file is None or not Path(tsv_file).exists():
        return adata

    tsv_path = Path(tsv_file)

    if tsv_path.suffix == '.json':
        # Store JSON as a serialized string for h5ad compatibility
        with open(tsv_path) as fh:
            data = json.load(fh)
        adata.uns[key] = json.dumps(data)
        logger.info(
            f"Added uns['{key}']: JSON from {tsv_file}.")
    else:
        try:
            df = pd.read_csv(tsv_file, sep='\t')
        except pd.errors.EmptyDataError:
            logger.info(
                f"Skipping uns['{key}']: empty file {tsv_file}.")
            return adata
        if len(df) == 0:
            logger.info(
                f"Skipping uns['{key}']: no data rows in {tsv_file}.")
            return adata
        # Ensure all columns have consistent types for h5ad
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str)
        adata.uns[key] = df
        logger.info(
            f"Added uns['{key}']: {df.shape[0]} rows x {df.shape[1]} cols "
            f"from {tsv_file}.")
    return adata


def main(args):
    """Run AnnData export pipeline."""
    import scanpy as sc

    logger = get_named_logger("Export")
    logger.info("Starting AnnData export.")

    # Load primary gene expression matrix
    logger.info(f"Loading gene matrix from {args.gene_matrix_dir}.")
    adata = sc.read_10x_mtx(
        str(args.gene_matrix_dir), var_names='gene_symbols')
    adata.var_names_make_unique()
    logger.info(
        f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} genes.")

    # Store raw counts
    adata.layers['counts'] = adata.X.copy()

    # Load transcript matrix as a secondary layer if available
    if (args.transcript_matrix_dir is not None
            and Path(args.transcript_matrix_dir).exists()):
        logger.info(
            f"Loading transcript matrix from "
            f"{args.transcript_matrix_dir}.")
        adata_tx = sc.read_10x_mtx(
            str(args.transcript_matrix_dir), var_names='gene_symbols')
        adata_tx.var_names_make_unique()
        adata.uns['transcript_var_names'] = list(adata_tx.var_names)
        adata.uns['n_transcripts'] = adata_tx.shape[1]
        logger.info(
            f"Transcript matrix: {adata_tx.shape[0]} cells x "
            f"{adata_tx.shape[1]} transcripts.")

    # Add cluster assignments
    cluster_files = {
        'gene_cluster': args.gene_clusters,
        'isoform_cluster': args.isoform_clusters,
        'joint_cluster': args.joint_clusters,
    }
    adata = load_and_merge_clusters(adata, cluster_files)

    # Add UMAP embedding
    adata = add_umap_embedding(adata, args.joint_umap, key='X_umap')

    # Add cell type annotations
    if args.cell_type_annotations is not None:
        adata = add_cell_metadata(
            adata, args.cell_type_annotations,
            columns=['cell_type', 'cluster'])

    # Add isoform diversity as per-cell metadata
    if args.isoform_diversity is not None:
        adata = add_cell_metadata(adata, args.isoform_diversity)

    # Add pseudotime
    if args.pseudotime is not None:
        adata = add_cell_metadata(
            adata, args.pseudotime,
            columns=['dpt_pseudotime'])

    # Add analysis result DataFrames to uns
    uns_data = {
        'dtu_results': args.dtu_results,
        'switching_results': args.switching_results,
        'novel_catalog': args.novel_catalog,
        'novel_enrichment': args.novel_enrichment,
        'cluster_comparison': args.cluster_comparison,
        'ase_results': args.ase_results,
    }
    for key, path in uns_data.items():
        adata = add_uns_dataframe(adata, key, path)

    # Write h5ad
    logger.info(f"Writing AnnData to {args.output}.")
    adata.write_h5ad(args.output)

    # Report
    n_obs_cols = len(adata.obs.columns)
    n_obsm_keys = len(adata.obsm)
    n_uns_keys = len(adata.uns)
    logger.info(
        f"Export complete: {adata.shape[0]} cells x {adata.shape[1]} genes, "
        f"{n_obs_cols} obs columns, {n_obsm_keys} obsm embeddings, "
        f"{n_uns_keys} uns entries -> {args.output}.")
