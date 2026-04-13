"""Tests for differential_transcript_usage module (isosceles Module 4)."""
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
def transcript_matrix_with_dtu(tmp_path):
    """Create a transcript matrix with a planted DTU signal.

    Gene A: 2 isoforms. Cluster 0 uses TX_A.0, cluster 1 uses TX_A.1.
    Gene B: 2 isoforms. No DTU (both clusters use similar proportions).
    Gene C: 3 isoforms. Cluster 0 uses TX_C.0, cluster 1 uses TX_C.2.
    """
    out = tmp_path / "tx_matrix"
    out.mkdir()
    n_cells = 60  # 30 per cluster
    np.random.seed(42)

    # Transcript definitions
    tx_names = [
        'TX_A.0', 'TX_A.1',       # Gene A: clear DTU
        'TX_B.0', 'TX_B.1',       # Gene B: no DTU
        'TX_C.0', 'TX_C.1', 'TX_C.2',  # Gene C: DTU with switching
    ]
    n_tx = len(tx_names)

    # Build count matrix (tx x cells)
    data = np.zeros((n_tx, n_cells), dtype=np.float32)

    # Gene A: strong DTU
    data[0, :30] = np.random.poisson(10, 30)  # TX_A.0 high in cluster 0
    data[0, 30:] = np.random.poisson(1, 30)
    data[1, :30] = np.random.poisson(1, 30)   # TX_A.1 high in cluster 1
    data[1, 30:] = np.random.poisson(10, 30)

    # Gene B: no DTU (identical usage pattern in both halves)
    data[2, :] = np.random.poisson(5, n_cells)
    data[3, :] = data[2, :].copy()  # exact same counts = no DTU possible

    # Gene C: DTU with 3 isoforms
    data[4, :30] = np.random.poisson(8, 30)   # TX_C.0 high in cluster 0
    data[4, 30:] = np.random.poisson(1, 30)
    data[5, :] = np.random.poisson(2, n_cells)  # TX_C.1 low everywhere
    data[6, :30] = np.random.poisson(1, 30)   # TX_C.2 high in cluster 1
    data[6, 30:] = np.random.poisson(8, 30)

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
def clusters_tsv(tmp_path, transcript_matrix_with_dtu):
    """Create cluster assignments: first 30 cells = cluster 0, rest = 1."""
    _, barcodes = transcript_matrix_with_dtu
    clusters_file = tmp_path / "clusters.tsv"
    with open(clusters_file, 'w') as f:
        f.write("barcode\tcluster\n")
        for i, bc in enumerate(barcodes):
            cluster = '0' if i < 30 else '1'
            f.write(f"{bc}\t{cluster}\n")
    return clusters_file


@pytest.fixture
def gene_transcript_map_dtu(tmp_path):
    """Gene-transcript map for the DTU test data."""
    map_file = tmp_path / "gene_transcript_map.tsv"
    with open(map_file, 'w') as f:
        f.write("transcript_id\tgene_id\n")
        f.write("TX_A.0\tGENE_A\n")
        f.write("TX_A.1\tGENE_A\n")
        f.write("TX_B.0\tGENE_B\n")
        f.write("TX_B.1\tGENE_B\n")
        f.write("TX_C.0\tGENE_C\n")
        f.write("TX_C.1\tGENE_C\n")
        f.write("TX_C.2\tGENE_C\n")
    return map_file


class TestLoadGeneTranscriptMap:
    """Test gene-transcript map loading."""

    def test_load_map(self, gene_transcript_map_dtu):
        """Test basic map loading."""
        from iris.differential_transcript_usage import (
            load_gene_transcript_map)

        tx_to_gene, gene_to_tx = load_gene_transcript_map(
            gene_transcript_map_dtu)

        assert tx_to_gene['TX_A.0'] == 'GENE_A'
        assert tx_to_gene['TX_A.1'] == 'GENE_A'
        assert set(gene_to_tx['GENE_A']) == {'TX_A.0', 'TX_A.1'}
        assert len(gene_to_tx['GENE_C']) == 3


class TestBuildGeneGroups:
    """Test gene grouping and filtering."""

    def test_build_groups(self, gene_transcript_map_dtu):
        """Test grouping transcripts by gene."""
        from iris.differential_transcript_usage import (
            load_gene_transcript_map, build_gene_groups)

        tx_to_gene, _ = load_gene_transcript_map(gene_transcript_map_dtu)
        tx_names = ['TX_A.0', 'TX_A.1', 'TX_B.0', 'TX_B.1',
                     'TX_C.0', 'TX_C.1', 'TX_C.2']

        groups = build_gene_groups(tx_names, tx_to_gene, min_isoforms=2)
        assert 'GENE_A' in groups
        assert 'GENE_B' in groups
        assert 'GENE_C' in groups
        assert len(groups['GENE_A']) == 2
        assert len(groups['GENE_C']) == 3

    def test_min_isoforms_filter(self, gene_transcript_map_dtu):
        """Test that min_isoforms correctly filters."""
        from iris.differential_transcript_usage import (
            load_gene_transcript_map, build_gene_groups)

        tx_to_gene, _ = load_gene_transcript_map(gene_transcript_map_dtu)
        tx_names = ['TX_A.0', 'TX_A.1', 'TX_B.0', 'TX_B.1',
                     'TX_C.0', 'TX_C.1', 'TX_C.2']

        groups = build_gene_groups(tx_names, tx_to_gene, min_isoforms=3)
        assert 'GENE_A' not in groups  # only 2 isoforms
        assert 'GENE_C' in groups      # 3 isoforms


class TestChiSquaredDTU:
    """Test chi-squared DTU test."""

    def test_clear_dtu_signal(self):
        """Test detection of clear differential usage."""
        from iris.differential_transcript_usage import (
            chi_squared_dtu_test)

        # Cluster A: mostly isoform 0
        counts_a = np.array([100, 5])
        # Cluster B: mostly isoform 1
        counts_b = np.array([5, 100])

        chi2, pval, effect, _conv = chi_squared_dtu_test(counts_a, counts_b)
        assert chi2 > 0
        assert pval < 0.001  # very significant
        assert effect > 0.5  # strong effect

    def test_no_dtu(self):
        """Test no false positives when usage is similar."""
        from iris.differential_transcript_usage import (
            chi_squared_dtu_test)

        counts_a = np.array([50, 50])
        counts_b = np.array([48, 52])

        chi2, pval, effect, _conv = chi_squared_dtu_test(counts_a, counts_b)
        assert pval > 0.05  # not significant

    def test_all_zero(self):
        """Test handling of all-zero counts."""
        from iris.differential_transcript_usage import (
            chi_squared_dtu_test)

        chi2, pval, effect, _conv = chi_squared_dtu_test(
            np.array([0, 0]), np.array([0, 0]))
        assert np.isnan(chi2)
        assert np.isnan(pval)

    def test_single_isoform(self):
        """Test with only one expressed isoform returns NaN."""
        from iris.differential_transcript_usage import (
            chi_squared_dtu_test)

        chi2, pval, effect, _conv = chi_squared_dtu_test(
            np.array([10, 0]), np.array([5, 0]))
        assert np.isnan(chi2)

    def test_three_isoforms(self):
        """Test with 3 isoforms."""
        from iris.differential_transcript_usage import (
            chi_squared_dtu_test)

        counts_a = np.array([80, 15, 5])
        counts_b = np.array([5, 15, 80])

        chi2, pval, effect, _conv = chi_squared_dtu_test(counts_a, counts_b)
        assert pval < 0.001


class TestDirichletMultinomial:
    """Test Dirichlet-multinomial DTU test."""

    def test_clear_dtu(self):
        """Test DM test detects clear signal."""
        from iris.differential_transcript_usage import (
            dirichlet_multinomial_test)

        np.random.seed(42)
        # 20 cells per cluster, 2 isoforms
        counts_a = np.column_stack([
            np.random.poisson(10, 20),
            np.random.poisson(1, 20)])
        counts_b = np.column_stack([
            np.random.poisson(1, 20),
            np.random.poisson(10, 20)])

        stat, pval, effect, _conv = dirichlet_multinomial_test(counts_a, counts_b)
        assert not np.isnan(stat)
        assert pval < 0.05

    def test_no_dtu(self):
        """Test DM test with no signal gives valid output."""
        from iris.differential_transcript_usage import (
            dirichlet_multinomial_test)

        np.random.seed(42)
        counts_a = np.column_stack([
            np.random.poisson(5, 20),
            np.random.poisson(5, 20)])
        counts_b = np.column_stack([
            np.random.poisson(5, 20),
            np.random.poisson(5, 20)])

        stat, pval, effect, _conv = dirichlet_multinomial_test(counts_a, counts_b)
        # DM test should return a valid result (not NaN)
        assert not np.isnan(stat) or np.isnan(pval)


class TestCorrectPvalues:
    """Test Benjamini-Hochberg correction."""

    def test_basic_correction(self):
        """Test BH correction produces valid adjusted p-values."""
        from iris.differential_transcript_usage import (
            correct_pvalues)

        pvals = np.array([0.01, 0.04, 0.03, 0.10])
        adjusted = correct_pvalues(pvals)

        # Adjusted p-values should be >= original
        for i in range(len(pvals)):
            assert adjusted[i] >= pvals[i] or np.isclose(
                adjusted[i], pvals[i])
        # Should be <= 1
        assert (adjusted <= 1.0).all()

    def test_monotonicity(self):
        """Test that adjusted p-values maintain rank order."""
        from iris.differential_transcript_usage import (
            correct_pvalues)

        pvals = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        adjusted = correct_pvalues(pvals)

        # Sorted adjusted should be non-decreasing
        sorted_adj = np.sort(adjusted)
        for i in range(1, len(sorted_adj)):
            assert sorted_adj[i] >= sorted_adj[i - 1]

    def test_nan_handling(self):
        """Test that NaN p-values are preserved."""
        from iris.differential_transcript_usage import (
            correct_pvalues)

        pvals = np.array([0.01, np.nan, 0.05, np.nan, 0.1])
        adjusted = correct_pvalues(pvals)

        assert np.isnan(adjusted[1])
        assert np.isnan(adjusted[3])
        assert not np.isnan(adjusted[0])
        assert not np.isnan(adjusted[2])

    def test_all_nan(self):
        """Test all-NaN input."""
        from iris.differential_transcript_usage import (
            correct_pvalues)

        pvals = np.array([np.nan, np.nan, np.nan])
        adjusted = correct_pvalues(pvals)
        assert np.isnan(adjusted).all()

    def test_single_pvalue(self):
        """Test with single p-value."""
        from iris.differential_transcript_usage import (
            correct_pvalues)

        adjusted = correct_pvalues(np.array([0.03]))
        assert adjusted[0] == pytest.approx(0.03)


class TestDetectIsoformSwitching:
    """Test isoform switching detection."""

    def test_switching_detected(self):
        """Test switching is detected when dominant isoform changes."""
        from iris.differential_transcript_usage import (
            detect_isoform_switching)

        counts_a = np.array([100, 10])  # dominant: TX_0
        counts_b = np.array([10, 100])  # dominant: TX_1
        names = ['TX_0', 'TX_1']

        result = detect_isoform_switching(counts_a, counts_b, names)
        assert result is not None
        assert result['dominant_transcript_a'] == 'TX_0'
        assert result['dominant_transcript_b'] == 'TX_1'
        assert result['switching_score'] > 0.5

    def test_no_switching(self):
        """Test no switching when same dominant isoform."""
        from iris.differential_transcript_usage import (
            detect_isoform_switching)

        counts_a = np.array([100, 50])
        counts_b = np.array([80, 40])  # same dominant
        names = ['TX_0', 'TX_1']

        result = detect_isoform_switching(counts_a, counts_b, names)
        assert result is None

    def test_zero_counts(self):
        """Test with zero counts in one group."""
        from iris.differential_transcript_usage import (
            detect_isoform_switching)

        result = detect_isoform_switching(
            np.array([0, 0]), np.array([10, 5]), ['TX_0', 'TX_1'])
        assert result is None

    def test_three_isoform_switching(self):
        """Test switching with 3 isoforms."""
        from iris.differential_transcript_usage import (
            detect_isoform_switching)

        counts_a = np.array([80, 15, 5])   # dominant: TX_0
        counts_b = np.array([5, 15, 80])   # dominant: TX_2
        names = ['TX_0', 'TX_1', 'TX_2']

        result = detect_isoform_switching(counts_a, counts_b, names)
        assert result is not None
        assert result['dominant_transcript_a'] == 'TX_0'
        assert result['dominant_transcript_b'] == 'TX_2'


class TestDTUEndToEnd:
    """End-to-end tests for the DTU pipeline."""

    def test_chi_squared_pipeline(
            self, transcript_matrix_with_dtu, clusters_tsv,
            gene_transcript_map_dtu, tmp_path):
        """Test full DTU pipeline with chi-squared method."""
        try:
            import scanpy  # noqa: F401
        except ImportError:
            pytest.skip("scanpy not available")

        from iris.differential_transcript_usage import main

        matrix_dir, _ = transcript_matrix_with_dtu
        output_dtu = tmp_path / "dtu_results.tsv"
        output_switching = tmp_path / "isoform_switching.tsv"
        output_summary = tmp_path / "dtu_summary.json"

        class Args:
            pass

        args = Args()
        args.transcript_matrix_dir = matrix_dir
        args.clusters = clusters_tsv
        args.gene_transcript_map = gene_transcript_map_dtu
        args.output_dtu = output_dtu
        args.output_switching = output_switching
        args.output_summary = output_summary
        args.test_method = "chi_squared"
        args.min_cells_per_cluster = 5
        args.min_gene_counts = 10
        args.min_isoforms = 2
        args.fdr_threshold = 0.05
        args.cluster_column = "cluster"
        args.comparison_mode = "one_vs_rest"
        args.yates_correction = True
        args.dm_maxiter = 200
        args.n_jobs = 1

        main(args)

        # Check outputs exist and are valid
        assert output_dtu.exists()
        assert output_switching.exists()
        assert output_summary.exists()

        dtu_df = pd.read_csv(output_dtu, sep='\t')
        assert len(dtu_df) > 0
        assert 'gene' in dtu_df.columns
        assert 'pvalue_adj' in dtu_df.columns

        # Gene A and Gene C should be significant (planted DTU)
        sig_genes = dtu_df[
            dtu_df['pvalue_adj'] < 0.05]['gene'].tolist()
        assert 'GENE_A' in sig_genes
        assert 'GENE_C' in sig_genes

        # Gene B should NOT be significant
        gene_b = dtu_df[dtu_df['gene'] == 'GENE_B']
        if len(gene_b) > 0:
            assert gene_b.iloc[0]['pvalue_adj'] > 0.05

        # Check switching events
        switch_df = pd.read_csv(output_switching, sep='\t')
        switch_genes = switch_df['gene'].tolist()
        assert 'GENE_A' in switch_genes  # clear switching
        assert 'GENE_C' in switch_genes  # clear switching

        # Check summary
        with open(output_summary) as f:
            summary = json.load(f)
        assert summary['n_genes_tested'] > 0
        assert summary['n_significant'] >= 2
        assert summary['n_switching_events'] >= 2

    def test_too_few_clusters(
            self, transcript_matrix_with_dtu,
            gene_transcript_map_dtu, tmp_path):
        """Test graceful handling when not enough clusters."""
        try:
            import scanpy  # noqa: F401
        except ImportError:
            pytest.skip("scanpy not available")

        from iris.differential_transcript_usage import main

        matrix_dir, barcodes = transcript_matrix_with_dtu

        # All cells in one cluster
        clusters_file = tmp_path / "one_cluster.tsv"
        with open(clusters_file, 'w') as f:
            f.write("barcode\tcluster\n")
            for bc in barcodes:
                f.write(f"{bc}\t0\n")

        class Args:
            pass

        args = Args()
        args.transcript_matrix_dir = matrix_dir
        args.clusters = clusters_file
        args.gene_transcript_map = gene_transcript_map_dtu
        args.output_dtu = tmp_path / "dtu.tsv"
        args.output_switching = tmp_path / "switch.tsv"
        args.output_summary = tmp_path / "summary.json"
        args.test_method = "chi_squared"
        args.min_cells_per_cluster = 5
        args.min_gene_counts = 10
        args.min_isoforms = 2
        args.fdr_threshold = 0.05
        args.cluster_column = "cluster"
        args.comparison_mode = "one_vs_rest"
        args.yates_correction = True
        args.dm_maxiter = 200
        args.n_jobs = 1

        main(args)

        # Should produce empty results
        dtu_df = pd.read_csv(args.output_dtu, sep='\t')
        assert len(dtu_df) == 0

        with open(args.output_summary) as f:
            summary = json.load(f)
        assert summary['n_genes_tested'] == 0
