"""Tests for cluster_analysis module."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import scipy.io
import scipy.sparse


@pytest.fixture
def mex_matrix_dir(tmp_path):
    """Create a minimal MEX format matrix directory."""
    n_cells = 50
    n_genes = 100
    np.random.seed(42)

    # Sparse count matrix: MEX format is genes x cells (rows=genes, cols=cells)
    data = np.random.poisson(2, size=(n_genes, n_cells)).astype(np.float32)
    sparse_mat = scipy.sparse.csc_matrix(data)
    scipy.io.mmwrite(str(tmp_path / "matrix.mtx"), sparse_mat)

    # Barcodes
    barcodes = [f"CELL{i:04d}-1" for i in range(n_cells)]
    with open(tmp_path / "barcodes.tsv", "w") as f:
        for bc in barcodes:
            f.write(f"{bc}\n")

    # Features/genes
    genes = [f"GENE{i:04d}" for i in range(n_genes)]
    with open(tmp_path / "features.tsv", "w") as f:
        for g in genes:
            f.write(f"{g}\t{g}\tGene Expression\n")

    # Compress files (scanpy expects .gz for read_10x_mtx or plain)
    import gzip
    import shutil
    for fname in ["matrix.mtx", "barcodes.tsv", "features.tsv"]:
        with open(tmp_path / fname, 'rb') as f_in:
            with gzip.open(tmp_path / f"{fname}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        (tmp_path / fname).unlink()

    return tmp_path


class TestClusterAnalysisHelpers:
    """Test helper functions in cluster_analysis."""

    def test_sctransform_normalize(self, mex_matrix_dir):
        """Test SCTransform-style normalization."""
        try:
            import scanpy as sc
            from iris.cluster_analysis import (
                load_mex_to_anndata, sctransform_normalize)

            adata = load_mex_to_anndata(mex_matrix_dir)
            adata = sctransform_normalize(adata)
            assert 'counts' in adata.layers
            assert adata.X is not None
        except ImportError:
            pytest.skip("scanpy not available")

    def test_scanpy_normalize(self, mex_matrix_dir):
        """Test standard Scanpy normalization."""
        try:
            import scanpy as sc
            from iris.cluster_analysis import (
                load_mex_to_anndata, scanpy_normalize)

            adata = load_mex_to_anndata(mex_matrix_dir)
            adata = scanpy_normalize(adata, target_sum=10000)
            assert 'counts' in adata.layers
        except ImportError:
            pytest.skip("scanpy not available")

    def test_cellranger_cell_calling(self, mex_matrix_dir):
        """Test Cell Ranger-style cell calling."""
        try:
            import scanpy as sc
            from iris.cluster_analysis import (
                load_mex_to_anndata, cellranger_cell_calling)

            adata = load_mex_to_anndata(mex_matrix_dir)
            n_before = adata.shape[0]
            adata = cellranger_cell_calling(adata, expected_cells=20)
            # Should retain some cells
            assert adata.shape[0] > 0
            assert adata.shape[0] <= n_before
        except ImportError:
            pytest.skip("scanpy not available")
