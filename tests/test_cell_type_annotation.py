"""Tests for cell_type_annotation module (isosceles Module 7)."""
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io
import scipy.sparse


@pytest.fixture
def gene_matrix_dir(tmp_path):
    """Create a gene matrix with marker gene names for annotation testing.

    Cluster 0 (first 30 cells): high T cell markers (CD3D, CD3E, CD4, CD8A).
    Cluster 1 (last 30 cells): high B cell markers (CD19, MS4A1, CD79A).
    Remaining genes are generic low-expression background.
    """
    out = tmp_path / "gene_matrix"
    out.mkdir()
    n_cells = 60
    np.random.seed(42)

    marker_genes = [
        'CD3D', 'CD3E', 'CD4', 'CD8A',   # T cell markers (indices 0-3)
        'CD19', 'MS4A1', 'CD79A',          # B cell markers (indices 4-6)
    ]
    background_genes = [f"GENE{i:04d}" for i in range(93)]
    gene_names = marker_genes + background_genes
    n_genes = len(gene_names)

    data = np.zeros((n_genes, n_cells), dtype=np.float32)

    # T cell markers: high in cluster 0
    for i in range(4):
        data[i, :30] = np.random.poisson(15, 30)
        data[i, 30:] = np.random.poisson(1, 30)

    # B cell markers: high in cluster 1
    for i in range(4, 7):
        data[i, :30] = np.random.poisson(1, 30)
        data[i, 30:] = np.random.poisson(15, 30)

    # Background genes: low random expression
    for i in range(7, n_genes):
        data[i, :] = np.random.poisson(2, n_cells)

    sparse_mat = scipy.sparse.csc_matrix(data)
    scipy.io.mmwrite(str(out / "matrix.mtx"), sparse_mat)

    barcodes = [f"CELL{i:04d}-1" for i in range(n_cells)]
    with open(out / "barcodes.tsv", "w") as f:
        for bc in barcodes:
            f.write(f"{bc}\n")
    with open(out / "features.tsv", "w") as f:
        for g in gene_names:
            f.write(f"{g}\t{g}\tGene Expression\n")
    for fname in ["matrix.mtx", "barcodes.tsv", "features.tsv"]:
        with open(out / fname, 'rb') as f_in:
            with gzip.open(out / f"{fname}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        (out / fname).unlink()

    return out, barcodes


@pytest.fixture
def clusters_tsv(tmp_path, gene_matrix_dir):
    """Cluster assignments: first 30 = cluster 0, rest = cluster 1."""
    _, barcodes = gene_matrix_dir
    f = tmp_path / "clusters.tsv"
    with open(f, 'w') as fh:
        fh.write("barcode\tcluster\n")
        for i, bc in enumerate(barcodes):
            fh.write(f"{bc}\t{'0' if i < 30 else '1'}\n")
    return f


class TestGetDefaultMarkers:
    """Test default marker gene database retrieval."""

    def test_human_markers(self):
        """Human markers contain expected T cell genes."""
        from iris.cell_type_annotation import get_default_markers

        markers = get_default_markers('human')
        assert 'CD4 T cells' in markers
        assert 'CD3D' in markers['CD4 T cells']
        assert len(markers) > 5

    def test_mouse_markers(self):
        """Mouse markers use proper casing (Cd3d not CD3D)."""
        from iris.cell_type_annotation import get_default_markers

        markers = get_default_markers('mouse')
        assert 'CD4 T cells' in markers
        assert 'Cd3d' in markers['CD4 T cells']
        assert 'CD3D' not in markers['CD4 T cells']


class TestLoadCustomMarkers:
    """Test custom marker gene TSV loading."""

    def test_valid_tsv(self, tmp_path):
        """Test loading a valid custom marker TSV."""
        from iris.cell_type_annotation import load_custom_markers

        tsv = tmp_path / "custom_markers.tsv"
        with open(tsv, 'w') as f:
            f.write("cell_type\tgene\n")
            f.write("T cells\tCD3D\n")
            f.write("T cells\tCD3E\n")
            f.write("B cells\tCD19\n")

        markers = load_custom_markers(tsv)
        assert 'T cells' in markers
        assert 'CD3D' in markers['T cells']
        assert len(markers['T cells']) == 2
        assert len(markers['B cells']) == 1

    def test_bad_columns(self, tmp_path):
        """Test that missing required columns raises ValueError."""
        from iris.cell_type_annotation import load_custom_markers

        tsv = tmp_path / "bad_markers.tsv"
        with open(tsv, 'w') as f:
            f.write("type\tmarker\n")
            f.write("T cells\tCD3D\n")

        with pytest.raises(ValueError, match="cell_type"):
            load_custom_markers(tsv)


class TestAnnotateByOverlap:
    """Test marker gene overlap annotation method."""

    def test_cluster_annotation(self, gene_matrix_dir, clusters_tsv):
        """Cluster 0 should be annotated as T cells."""
        try:
            import scanpy as sc
        except ImportError:
            pytest.skip("scanpy not available")

        from iris.cell_type_annotation import (
            annotate_clusters_by_overlap, get_default_markers)

        matrix_path, _ = gene_matrix_dir
        adata = sc.read_10x_mtx(str(matrix_path), var_names='gene_symbols')
        adata.var_names_make_unique()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        clusters_df = pd.read_csv(clusters_tsv, sep='\t')
        cluster_labels = pd.Series(
            clusters_df['cluster'].astype(str).values,
            index=clusters_df['barcode'].values)

        marker_db = get_default_markers('human')
        result = annotate_clusters_by_overlap(
            adata, cluster_labels, marker_db, min_markers=3)

        assert len(result) == 2
        cl0 = result[result['cluster'] == '0']
        assert 'T cells' in cl0.iloc[0]['cell_type']

    def test_min_marker_filter(self, gene_matrix_dir, clusters_tsv):
        """High min_markers threshold should yield Unknown."""
        try:
            import scanpy as sc
        except ImportError:
            pytest.skip("scanpy not available")

        from iris.cell_type_annotation import (
            annotate_clusters_by_overlap, get_default_markers)

        matrix_path, _ = gene_matrix_dir
        adata = sc.read_10x_mtx(str(matrix_path), var_names='gene_symbols')
        adata.var_names_make_unique()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        clusters_df = pd.read_csv(clusters_tsv, sep='\t')
        cluster_labels = pd.Series(
            clusters_df['cluster'].astype(str).values,
            index=clusters_df['barcode'].values)

        marker_db = get_default_markers('human')
        result = annotate_clusters_by_overlap(
            adata, cluster_labels, marker_db, min_markers=50)

        # All clusters should be Unknown with impossibly high threshold
        assert (result['cell_type'] == 'Unknown').all()


class TestAnnotateByCorrelation:
    """Test correlation-based annotation method."""

    def test_correlation_assigns_types(self, gene_matrix_dir, clusters_tsv):
        """Correlation method should assign cell types to clusters."""
        try:
            import scanpy as sc
        except ImportError:
            pytest.skip("scanpy not available")

        from iris.cell_type_annotation import (
            annotate_clusters_by_correlation, get_default_markers)

        matrix_path, _ = gene_matrix_dir
        adata = sc.read_10x_mtx(str(matrix_path), var_names='gene_symbols')
        adata.var_names_make_unique()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        clusters_df = pd.read_csv(clusters_tsv, sep='\t')
        cluster_labels = pd.Series(
            clusters_df['cluster'].astype(str).values,
            index=clusters_df['barcode'].values)

        marker_db = get_default_markers('human')
        result = annotate_clusters_by_correlation(
            adata, cluster_labels, marker_db)

        assert len(result) == 2
        assert 'correlation' in result.columns
        assert 'pvalue' in result.columns
        # At least one cluster should be annotated (not Unknown)
        assert (result['cell_type'] != 'Unknown').any()


class TestEndToEnd:
    """End-to-end test of the cell type annotation pipeline."""

    def test_full_pipeline(self, gene_matrix_dir, clusters_tsv, tmp_path):
        """Test full pipeline produces valid outputs."""
        try:
            import scanpy  # noqa: F401
        except ImportError:
            pytest.skip("scanpy not available")

        from iris.cell_type_annotation import main

        matrix_path, _ = gene_matrix_dir

        class Args:
            pass

        args = Args()
        args.gene_matrix_dir = matrix_path
        args.clusters = clusters_tsv
        args.marker_genes_db = None
        args.output_annotations = tmp_path / "annotations.tsv"
        args.output_cluster_types = tmp_path / "cluster_types.tsv"
        args.output_summary = tmp_path / "summary.json"
        args.method = "marker_overlap"
        args.min_marker_genes = 3
        args.cluster_column = "cluster"
        args.species = "human"

        main(args)

        assert args.output_annotations.exists()
        assert args.output_cluster_types.exists()
        assert args.output_summary.exists()

        cluster_types = pd.read_csv(
            args.output_cluster_types, sep='\t')
        assert len(cluster_types) == 2

        annotations = pd.read_csv(args.output_annotations, sep='\t')
        assert len(annotations) == 60
        assert 'cell_type' in annotations.columns

        with open(args.output_summary) as f:
            summary = json.load(f)
        assert summary['n_clusters'] == 2
        assert summary['n_annotated_clusters'] >= 1
