"""Tests for export_anndata module (isosceles Module 10)."""
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
    """Create a minimal gene-level MEX matrix directory."""
    out = tmp_path / "gene_matrix"
    out.mkdir()
    n_cells = 30
    n_genes = 50
    np.random.seed(42)

    data = np.random.poisson(3, size=(n_genes, n_cells)).astype(np.float32)
    sparse_mat = scipy.sparse.csc_matrix(data)
    scipy.io.mmwrite(str(out / "matrix.mtx"), sparse_mat)

    barcodes = [f"CELL{i:04d}-1" for i in range(n_cells)]
    with open(out / "barcodes.tsv", "w") as f:
        for bc in barcodes:
            f.write(f"{bc}\n")
    genes = [f"GENE{i:04d}" for i in range(n_genes)]
    with open(out / "features.tsv", "w") as f:
        for g in genes:
            f.write(f"{g}\t{g}\tGene Expression\n")
    for fname in ["matrix.mtx", "barcodes.tsv", "features.tsv"]:
        with open(out / fname, 'rb') as f_in:
            with gzip.open(out / f"{fname}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        (out / fname).unlink()

    return out, barcodes


class TestLoadAndMergeClusters:
    """Test cluster loading and merging into adata.obs."""

    def test_merge_clusters(self, tmp_path):
        """Test that cluster columns are added to adata.obs."""
        import anndata

        from iris.export_anndata import load_and_merge_clusters

        barcodes = [f"CELL{i:04d}-1" for i in range(20)]
        adata = anndata.AnnData(
            X=np.random.random((20, 10)).astype(np.float32),
            obs=pd.DataFrame(index=barcodes))

        clusters_file = tmp_path / "clusters.tsv"
        with open(clusters_file, 'w') as f:
            f.write("barcode\tcluster\n")
            for i, bc in enumerate(barcodes):
                f.write(f"{bc}\t{'A' if i < 10 else 'B'}\n")

        adata = load_and_merge_clusters(
            adata, {'gene_cluster': clusters_file})

        assert 'gene_cluster' in adata.obs.columns
        assert adata.obs.loc['CELL0000-1', 'gene_cluster'] == 'A'
        assert adata.obs.loc['CELL0015-1', 'gene_cluster'] == 'B'

    def test_missing_file_skipped(self, tmp_path):
        """Test that missing files are silently skipped."""
        import anndata

        from iris.export_anndata import load_and_merge_clusters

        adata = anndata.AnnData(
            X=np.random.random((5, 3)).astype(np.float32),
            obs=pd.DataFrame(index=[f"C{i}" for i in range(5)]))

        adata = load_and_merge_clusters(
            adata, {'missing_col': tmp_path / "nonexistent.tsv"})

        assert 'missing_col' not in adata.obs.columns


class TestAddUmapEmbedding:
    """Test UMAP embedding addition."""

    def test_add_umap(self, tmp_path):
        """Test UMAP coordinates are added to obsm."""
        import anndata

        from iris.export_anndata import add_umap_embedding

        barcodes = [f"CELL{i:04d}-1" for i in range(10)]
        adata = anndata.AnnData(
            X=np.random.random((10, 5)).astype(np.float32),
            obs=pd.DataFrame(index=barcodes))

        umap_file = tmp_path / "umap.tsv"
        with open(umap_file, 'w') as f:
            f.write("barcode\tD1\tD2\n")
            for bc in barcodes:
                f.write(f"{bc}\t{np.random.randn()}\t{np.random.randn()}\n")

        adata = add_umap_embedding(adata, umap_file, key='X_umap')

        assert 'X_umap' in adata.obsm
        assert adata.obsm['X_umap'].shape == (10, 2)


class TestAddCellMetadata:
    """Test per-cell metadata addition."""

    def test_add_metadata(self, tmp_path):
        """Test that metadata columns are added to obs."""
        import anndata

        from iris.export_anndata import add_cell_metadata

        barcodes = [f"CELL{i:04d}-1" for i in range(10)]
        adata = anndata.AnnData(
            X=np.random.random((10, 5)).astype(np.float32),
            obs=pd.DataFrame(index=barcodes))

        meta_file = tmp_path / "metadata.tsv"
        with open(meta_file, 'w') as f:
            f.write("barcode\tcell_type\tscore\n")
            for bc in barcodes:
                f.write(f"{bc}\tT cells\t0.95\n")

        adata = add_cell_metadata(adata, meta_file)

        assert 'cell_type' in adata.obs.columns
        assert 'score' in adata.obs.columns
        assert adata.obs.loc['CELL0000-1', 'cell_type'] == 'T cells'


class TestAddUnsDataframe:
    """Test uns DataFrame addition."""

    def test_add_uns(self, tmp_path):
        """Test that a DataFrame is stored in uns."""
        import anndata

        from iris.export_anndata import add_uns_dataframe

        adata = anndata.AnnData(
            X=np.random.random((5, 3)).astype(np.float32),
            obs=pd.DataFrame(index=[f"C{i}" for i in range(5)]))

        tsv_file = tmp_path / "results.tsv"
        with open(tsv_file, 'w') as f:
            f.write("gene\tpvalue\n")
            f.write("GENE_A\t0.01\n")
            f.write("GENE_B\t0.05\n")

        adata = add_uns_dataframe(adata, 'dtu_results', tsv_file)

        assert 'dtu_results' in adata.uns
        assert len(adata.uns['dtu_results']) == 2
        assert 'gene' in adata.uns['dtu_results'].columns


class TestEndToEnd:
    """End-to-end test of the AnnData export pipeline."""

    def test_full_pipeline(self, gene_matrix_dir, tmp_path):
        """Test full export produces valid h5ad."""
        try:
            import scanpy  # noqa: F401
        except ImportError:
            pytest.skip("scanpy not available")

        from iris.export_anndata import main

        matrix_path, barcodes = gene_matrix_dir

        # Create cluster file
        clusters_file = tmp_path / "clusters.tsv"
        with open(clusters_file, 'w') as f:
            f.write("barcode\tcluster\n")
            for i, bc in enumerate(barcodes):
                f.write(f"{bc}\t{'0' if i < 15 else '1'}\n")

        # Create UMAP file
        umap_file = tmp_path / "umap.tsv"
        np.random.seed(42)
        with open(umap_file, 'w') as f:
            f.write("barcode\tD1\tD2\n")
            for bc in barcodes:
                f.write(f"{bc}\t{np.random.randn()}\t{np.random.randn()}\n")

        output_h5ad = tmp_path / "output.h5ad"

        class Args:
            pass

        args = Args()
        args.gene_matrix_dir = matrix_path
        args.transcript_matrix_dir = None
        args.gene_clusters = clusters_file
        args.isoform_clusters = None
        args.joint_clusters = None
        args.joint_umap = umap_file
        args.cell_type_annotations = None
        args.isoform_diversity = None
        args.dtu_results = None
        args.switching_results = None
        args.novel_catalog = None
        args.novel_enrichment = None
        args.cluster_comparison = None
        args.pseudotime = None
        args.ase_results = None
        args.output = output_h5ad

        main(args)

        assert output_h5ad.exists()

        import anndata
        adata = anndata.read_h5ad(output_h5ad)
        assert adata.shape[0] == 30
        assert 'gene_cluster' in adata.obs.columns
        assert 'X_umap' in adata.obsm
        assert 'counts' in adata.layers
