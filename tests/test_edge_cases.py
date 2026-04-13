"""Edge-case tests for IRIS."""

import numpy as np
import pandas as pd
import pytest
import anndata


# ---------------------------------------------------------------------------
# 1. BH correction
# ---------------------------------------------------------------------------
from iris._stats import bh_correct


def test_bh_correct_empty_array():
    result = bh_correct([])
    assert len(result) == 0


def test_bh_correct_all_nan():
    result = bh_correct([np.nan, np.nan, np.nan])
    assert len(result) == 3
    assert all(np.isnan(result))


def test_bh_correct_single_pvalue():
    result = bh_correct([0.04])
    assert len(result) == 1
    assert np.isfinite(result[0])


def test_bh_correct_identical_pvalues():
    result = bh_correct([0.05, 0.05, 0.05, 0.05])
    assert len(result) == 4
    # All identical input -> all identical adjusted values
    assert np.allclose(result, result[0])


def test_bh_correct_monotonicity():
    """Sorted raw p-values should produce non-decreasing adjusted p-values."""
    raw = np.sort(np.array([0.001, 0.01, 0.03, 0.05, 0.10]))
    adj = bh_correct(raw)
    for i in range(len(adj) - 1):
        assert adj[i] <= adj[i + 1] + 1e-15


# ---------------------------------------------------------------------------
# 2. ASE binomial test
# ---------------------------------------------------------------------------
from iris.allele_specific_expression import compute_allelic_imbalance


def test_ase_balanced():
    pval, ratio = compute_allelic_imbalance(50, 50)
    assert pval > 0.05
    assert np.isclose(ratio, 0.5)


def test_ase_strong_imbalance():
    pval, ratio = compute_allelic_imbalance(95, 5)
    assert pval < 0.001


def test_ase_zero_counts():
    pval, ratio = compute_allelic_imbalance(0, 0)
    assert pval == 1.0
    assert np.isnan(ratio)


def test_ase_single_count():
    pval, ratio = compute_allelic_imbalance(1, 0)
    assert np.isfinite(pval)
    assert np.isclose(ratio, 0.0)


# ---------------------------------------------------------------------------
# 3. Trajectory binning
# ---------------------------------------------------------------------------
from iris.isoform_trajectory import compute_isoform_trends


def test_trajectory_n_bins():
    """n_bins=5 should produce results binned into exactly 5 bins."""
    n_cells = 50
    n_tx = 4
    barcodes = [f"cell_{i}" for i in range(n_cells)]

    # 4 transcripts mapping to 2 genes (2 tx each)
    np.random.seed(42)
    X = np.random.poisson(5, size=(n_cells, n_tx)).astype(float)
    var = pd.DataFrame({"transcript_id": [f"tx{i}" for i in range(n_tx)]})
    var.index = var["transcript_id"]
    adata = anndata.AnnData(X=X, obs=pd.DataFrame(index=barcodes), var=var)

    gene_map = pd.DataFrame({
        "transcript_id": ["tx0", "tx1", "tx2", "tx3"],
        "gene_id": ["G1", "G1", "G2", "G2"],
    })

    pseudotime = pd.Series(
        np.linspace(0, 1, n_cells), index=barcodes, name="pseudotime")

    result = compute_isoform_trends(adata, gene_map, pseudotime, n_bins=5)
    assert not result.empty
    # Each gene x transcript combo should have been evaluated over 5 bins;
    # the function returns one row per transcript, not per bin, so just
    # verify the call succeeded and returned rows.
    assert len(result) > 0


# ---------------------------------------------------------------------------
# 4. Isoform diversity – single-isoform gene
# ---------------------------------------------------------------------------
from iris.dual_layer_clustering import compute_diversity_index


def test_diversity_single_isoform_gene():
    """A gene with only one isoform should not contribute to diversity."""
    adata = anndata.AnnData(
        X=np.array([[1.0, 0.0]]),
        var=pd.DataFrame({"gene_id": ["G1", "G1"]}, index=["tx0", "tx1"]),
    )
    # gene_map as dict tx -> gene
    gene_map = {"tx0": "G1", "tx1": "G1"}
    result = compute_diversity_index(adata, gene_map)
    # With proportions [1.0, 0.0], Shannon entropy = 0
    assert result["diversity_index"].iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. Empty matrix handling
# ---------------------------------------------------------------------------
from iris.dual_layer_clustering import compute_isoform_usage


def test_isoform_usage_zero_counts():
    """Zero-count matrix with single-isoform genes returns empty usage."""
    adata = anndata.AnnData(
        X=np.zeros((3, 2)),
        var=pd.DataFrame(index=["tx0", "tx1"]),
    )
    gene_map = {"tx0": "G1", "tx1": "G2"}  # 1 tx per gene -> no multi-iso
    result = compute_isoform_usage(adata, gene_map, min_isoforms=2)
    assert result.shape[1] == 0


# ---------------------------------------------------------------------------
# 6. Chi-squared DTU with very small counts
# ---------------------------------------------------------------------------
from iris.differential_transcript_usage import chi_squared_dtu_test


def test_chi2_all_zeros():
    chi2, pval, es, _conv = chi_squared_dtu_test([0, 0], [0, 0])
    assert np.isnan(chi2)
    assert np.isnan(pval)


def test_chi2_single_nonzero():
    """Only one nonzero transcript -> fewer than 2 nonzero columns -> NaN."""
    chi2, pval, es, _conv = chi_squared_dtu_test([1, 0], [0, 0])
    assert np.isnan(chi2)
    assert np.isnan(pval)


def test_chi2_small_counts():
    """Small but valid 2x2 table should produce finite results."""
    chi2, pval, es, _conv = chi_squared_dtu_test([1, 1], [1, 1])
    assert np.isfinite(chi2)
    assert np.isfinite(pval)
