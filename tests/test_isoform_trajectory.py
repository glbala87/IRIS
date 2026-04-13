"""Tests for isoform_trajectory module (isosceles Module 8)."""
import gzip
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io
import scipy.sparse


@pytest.fixture
def transcript_matrix_with_trends(tmp_path):
    """Create a transcript matrix with planted pseudotime trends.

    TX_A.0 increases along pseudotime (high in late cells).
    TX_A.1 decreases along pseudotime (high in early cells).
    TX_B.0 and TX_B.1 are stable (no trend).
    """
    out = tmp_path / "tx_matrix"
    out.mkdir()
    n_cells = 60
    np.random.seed(42)

    tx_names = ['TX_A.0', 'TX_A.1', 'TX_B.0', 'TX_B.1']
    n_tx = len(tx_names)

    data = np.zeros((n_tx, n_cells), dtype=np.float32)

    # TX_A.0: increasing along cell index (proxy for pseudotime)
    for i in range(n_cells):
        data[0, i] = np.random.poisson(max(1, int(i * 0.5)))
    # TX_A.1: decreasing
    for i in range(n_cells):
        data[1, i] = np.random.poisson(max(1, int((n_cells - i) * 0.5)))
    # TX_B.0 and TX_B.1: stable
    data[2, :] = np.random.poisson(5, n_cells)
    data[3, :] = np.random.poisson(5, n_cells)

    sparse_mat = scipy.sparse.csc_matrix(data)
    scipy.io.mmwrite(str(out / "matrix.mtx"), sparse_mat)

    barcodes = [f"CELL{i:04d}-1" for i in range(n_cells)]
    with open(out / "barcodes.tsv", "w") as f:
        for bc in barcodes:
            f.write(f"{bc}\n")
    with open(out / "features.tsv", "w") as f:
        for tx in tx_names:
            f.write(f"{tx}\t{tx}\tGene Expression\n")
    for fname in ["matrix.mtx", "barcodes.tsv", "features.tsv"]:
        with open(out / fname, 'rb') as f_in:
            with gzip.open(out / f"{fname}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        (out / fname).unlink()

    return out, barcodes


@pytest.fixture
def gene_transcript_map(tmp_path):
    """Gene-transcript map for trend test data."""
    f = tmp_path / "map.tsv"
    with open(f, 'w') as fh:
        fh.write("transcript_id\tgene_id\n")
        fh.write("TX_A.0\tGENE_A\n")
        fh.write("TX_A.1\tGENE_A\n")
        fh.write("TX_B.0\tGENE_B\n")
        fh.write("TX_B.1\tGENE_B\n")
    return f


@pytest.fixture
def gene_matrix_dir_for_dpt(tmp_path):
    """Create a realistic gene matrix for diffusion pseudotime.

    Uses the negative binomial + dropout pattern from
    test_dual_layer_clustering to ensure HVG selection works.
    """
    out = tmp_path / "gene_matrix"
    out.mkdir()
    n_cells = 100
    n_genes = 200
    np.random.seed(42)

    data = np.random.negative_binomial(
        1, 0.8, size=(n_genes, n_cells)).astype(np.float32)
    dropout_mask = np.random.random((n_genes, n_cells)) < 0.6
    data[dropout_mask] = 0
    for g in range(30):
        data[g, :50] = np.random.negative_binomial(
            5, 0.2, 50).astype(np.float32)
        data[g, 50:] = np.random.negative_binomial(
            1, 0.9, 50).astype(np.float32)

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

    return out


class TestComputeIsoformTrends:
    """Test isoform proportion trend analysis."""

    def test_detects_trends(
            self, transcript_matrix_with_trends, gene_transcript_map):
        """Spearman correlation should detect planted trends."""
        try:
            import scanpy as sc
        except ImportError:
            pytest.skip("scanpy not available")

        from iris.isoform_trajectory import (
            compute_isoform_trends)

        matrix_path, barcodes = transcript_matrix_with_trends
        adata_tx = sc.read_10x_mtx(
            str(matrix_path), var_names='gene_symbols')

        gene_map = pd.read_csv(gene_transcript_map, sep='\t')

        # Create artificial pseudotime: linearly increasing with cell index
        pseudotime = pd.Series(
            np.linspace(0, 1, len(barcodes)),
            index=barcodes)

        trends = compute_isoform_trends(
            adata_tx, gene_map, pseudotime, n_bins=10)

        assert len(trends) > 0
        assert 'spearman_r' in trends.columns
        assert 'trend' in trends.columns

        # TX_A.0 should be increasing
        tx_a0 = trends[trends['transcript'] == 'TX_A.0']
        assert len(tx_a0) == 1
        assert tx_a0.iloc[0]['spearman_r'] > 0.3
        assert tx_a0.iloc[0]['trend'] == 'increasing'

        # TX_A.1 should be decreasing
        tx_a1 = trends[trends['transcript'] == 'TX_A.1']
        assert len(tx_a1) == 1
        assert tx_a1.iloc[0]['spearman_r'] < -0.3
        assert tx_a1.iloc[0]['trend'] == 'decreasing'

    def test_no_overlap(self, gene_transcript_map):
        """No overlapping barcodes should return empty DataFrame."""
        try:
            import scanpy as sc
        except ImportError:
            pytest.skip("scanpy not available")

        import anndata

        from iris.isoform_trajectory import (
            compute_isoform_trends)

        adata_tx = anndata.AnnData(
            X=np.random.random((5, 2)).astype(np.float32),
            obs=pd.DataFrame(index=[f"A{i}" for i in range(5)]),
            var=pd.DataFrame(index=['TX_A.0', 'TX_A.1']))

        gene_map = pd.read_csv(gene_transcript_map, sep='\t')
        pseudotime = pd.Series(
            np.linspace(0, 1, 5),
            index=[f"B{i}" for i in range(5)])  # no overlap

        trends = compute_isoform_trends(
            adata_tx, gene_map, pseudotime, n_bins=5)
        assert len(trends) == 0


class TestDetectTrajectorySwitching:
    """Test isoform switching detection along trajectory."""

    def test_switching_detected(self):
        """Opposing trends should produce a switching event."""
        from iris.isoform_trajectory import (
            detect_trajectory_switching)

        trends = pd.DataFrame([
            {'gene': 'GENE_A', 'transcript': 'TX_A.0',
             'spearman_r': 0.9, 'pvalue': 0.001, 'pvalue_adj': 0.005,
             'trend': 'increasing', 'mean_proportion': 0.6},
            {'gene': 'GENE_A', 'transcript': 'TX_A.1',
             'spearman_r': -0.85, 'pvalue': 0.002, 'pvalue_adj': 0.008,
             'trend': 'decreasing', 'mean_proportion': 0.4},
        ])

        switching = detect_trajectory_switching(trends, pval_threshold=0.05)
        assert len(switching) == 1
        assert switching.iloc[0]['gene'] == 'GENE_A'
        assert switching.iloc[0]['transcript_increasing'] == 'TX_A.0'
        assert switching.iloc[0]['transcript_decreasing'] == 'TX_A.1'
        assert switching.iloc[0]['switch_strength'] > 1.0

    def test_no_switching(self):
        """Same-direction trends should not produce switching."""
        from iris.isoform_trajectory import (
            detect_trajectory_switching)

        trends = pd.DataFrame([
            {'gene': 'GENE_A', 'transcript': 'TX_A.0',
             'spearman_r': 0.8, 'pvalue': 0.001, 'pvalue_adj': 0.003,
             'trend': 'increasing', 'mean_proportion': 0.6},
            {'gene': 'GENE_A', 'transcript': 'TX_A.1',
             'spearman_r': 0.6, 'pvalue': 0.01, 'pvalue_adj': 0.02,
             'trend': 'increasing', 'mean_proportion': 0.4},
        ])

        switching = detect_trajectory_switching(trends, pval_threshold=0.05)
        assert len(switching) == 0

    def test_empty_trends(self):
        """Empty trends DataFrame should return empty switching."""
        from iris.isoform_trajectory import (
            detect_trajectory_switching)

        switching = detect_trajectory_switching(pd.DataFrame())
        assert len(switching) == 0


class TestBHCorrect:
    """Test Benjamini-Hochberg correction."""

    def test_basic_correction(self):
        """Adjusted p-values should be >= original and <= 1."""
        from iris.isoform_trajectory import _bh_correct

        pvals = np.array([0.01, 0.04, 0.03, 0.10])
        adj = _bh_correct(pvals)
        assert (adj >= pvals).all()
        assert (adj <= 1.0).all()

    def test_nan_handling(self):
        """NaN p-values should be preserved."""
        from iris.isoform_trajectory import _bh_correct

        pvals = np.array([0.01, np.nan, 0.05])
        adj = _bh_correct(pvals)
        assert np.isnan(adj[1])
        assert not np.isnan(adj[0])
        assert not np.isnan(adj[2])


class TestComputeDiffusionPseudotime:
    """Test diffusion pseudotime computation."""

    def test_dpt_produces_pseudotime(self, gene_matrix_dir_for_dpt):
        """DPT should add dpt_pseudotime to obs."""
        try:
            import scanpy as sc
        except ImportError:
            pytest.skip("scanpy not available")

        from iris.isoform_trajectory import (
            compute_diffusion_pseudotime)

        adata = sc.read_10x_mtx(
            str(gene_matrix_dir_for_dpt), var_names='gene_symbols')
        adata.var_names_make_unique()

        result = compute_diffusion_pseudotime(
            adata, n_neighbors=15, n_pcs=20)

        assert 'dpt_pseudotime' in result.obs.columns
        pt = result.obs['dpt_pseudotime'].values
        assert not np.isnan(pt).all()
        assert pt.min() >= 0
        # Pseudotime should have variation
        assert pt.std() > 0

    def test_dpt_diffmap_embedding(self, gene_matrix_dir_for_dpt):
        """DPT should produce a diffusion map embedding."""
        try:
            import scanpy as sc
        except ImportError:
            pytest.skip("scanpy not available")

        from iris.isoform_trajectory import (
            compute_diffusion_pseudotime)

        adata = sc.read_10x_mtx(
            str(gene_matrix_dir_for_dpt), var_names='gene_symbols')
        adata.var_names_make_unique()

        result = compute_diffusion_pseudotime(
            adata, n_neighbors=15, n_pcs=20)

        assert 'X_diffmap' in result.obsm
        assert result.obsm['X_diffmap'].shape[0] == adata.shape[0]
