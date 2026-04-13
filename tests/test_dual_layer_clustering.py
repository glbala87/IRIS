"""Tests for dual_layer_clustering module (isosceles Module 3)."""
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
    n_cells = 100
    n_genes = 200
    np.random.seed(42)

    # Create realistic scRNA-seq-like data with dropout and variable genes
    # Base: sparse counts (most genes lowly expressed)
    data = np.random.negative_binomial(1, 0.8, size=(n_genes, n_cells)).astype(np.float32)
    # Add dropout (set ~60% to zero)
    dropout_mask = np.random.random((n_genes, n_cells)) < 0.6
    data[dropout_mask] = 0
    # Make 30 genes highly variable (high in half the cells, low in others)
    for g in range(30):
        data[g, :50] = np.random.negative_binomial(5, 0.2, 50).astype(np.float32)
        data[g, 50:] = np.random.negative_binomial(1, 0.9, 50).astype(np.float32)
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


@pytest.fixture
def transcript_matrix_dir(tmp_path):
    """Create a minimal transcript-level MEX matrix directory.

    Each gene has 2-3 transcripts, with different usage patterns
    across cells to create detectable isoform structure.
    """
    out = tmp_path / "transcript_matrix"
    out.mkdir()
    n_cells = 60
    np.random.seed(42)

    # 20 genes, each with 2-3 transcripts = ~50 transcripts
    transcript_names = []
    gene_for_transcript = []
    for g in range(20):
        n_tx = 2 if g < 10 else 3
        for t in range(n_tx):
            transcript_names.append(f"TX_GENE{g:04d}.{t}")
            gene_for_transcript.append(f"GENE{g:04d}")

    n_tx_total = len(transcript_names)

    # Create count data with isoform structure:
    # First 30 cells prefer isoform 0, last 30 prefer isoform 1
    data = np.zeros((n_tx_total, n_cells), dtype=np.float32)
    tx_idx = 0
    for g in range(20):
        n_tx = 2 if g < 10 else 3
        for t in range(n_tx):
            if t == 0:
                # High in first 30 cells
                data[tx_idx, :30] = np.random.poisson(5, 30)
                data[tx_idx, 30:] = np.random.poisson(1, 30)
            elif t == 1:
                # High in last 30 cells
                data[tx_idx, :30] = np.random.poisson(1, 30)
                data[tx_idx, 30:] = np.random.poisson(5, 30)
            else:
                # Low across all cells
                data[tx_idx, :] = np.random.poisson(1, n_cells)
            tx_idx += 1

    sparse_mat = scipy.sparse.csc_matrix(data)
    scipy.io.mmwrite(str(out / "matrix.mtx"), sparse_mat)

    barcodes = [f"CELL{i:04d}-1" for i in range(n_cells)]
    with open(out / "barcodes.tsv", "w") as f:
        for bc in barcodes:
            f.write(f"{bc}\n")

    with open(out / "features.tsv", "w") as f:
        for tx in transcript_names:
            f.write(f"{tx}\t{tx}\tGene Expression\n")

    for fname in ["matrix.mtx", "barcodes.tsv", "features.tsv"]:
        with open(out / fname, 'rb') as f_in:
            with gzip.open(out / f"{fname}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        (out / fname).unlink()

    return out, transcript_names, gene_for_transcript


@pytest.fixture
def transcript_matrix_path(transcript_matrix_dir):
    """Extract just the path from transcript_matrix_dir fixture."""
    path, _, _ = transcript_matrix_dir
    return path


@pytest.fixture
def gene_transcript_map(tmp_path, transcript_matrix_dir):
    """Create a gene-transcript mapping TSV."""
    map_file = tmp_path / "gene_transcript_map.tsv"
    _, names, genes = transcript_matrix_dir
    with open(map_file, 'w') as f:
        f.write("transcript_id\tgene_id\n")
        for tx, gene in zip(names, genes):
            f.write(f"{tx}\t{gene}\n")
    return map_file


class TestLoadGeneTranscriptMap:
    """Test gene-transcript map loading."""

    def test_load_valid_map(self, gene_transcript_map):
        """Test loading a valid mapping file."""
        from iris.dual_layer_clustering import (
            load_gene_transcript_map)

        gene_map = load_gene_transcript_map(gene_transcript_map)
        assert isinstance(gene_map, dict)
        assert len(gene_map) > 0
        # Check a known mapping
        assert gene_map['TX_GENE0000.0'] == 'GENE0000'
        assert gene_map['TX_GENE0000.1'] == 'GENE0000'

    def test_load_bad_map(self, tmp_path):
        """Test loading a single-column file raises ValueError."""
        from iris.dual_layer_clustering import (
            load_gene_transcript_map)

        bad_file = tmp_path / "bad_map.tsv"
        bad_file.write_text("only_one_column\nval1\nval2\n")
        with pytest.raises(ValueError, match="at least 2 columns"):
            load_gene_transcript_map(bad_file)


class TestComputeIsoformUsage:
    """Test isoform usage proportion computation."""

    def test_usage_proportions_sum_to_one(
            self, transcript_matrix_path, gene_transcript_map):
        """Verify that usage proportions sum to 1 per gene per cell."""
        try:
            from iris.dual_layer_clustering import (
                load_mex_to_anndata, load_gene_transcript_map,
                compute_isoform_usage)
        except ImportError:
            pytest.skip("scanpy not available")

        adata = load_mex_to_anndata(transcript_matrix_path)
        gene_map = load_gene_transcript_map(gene_transcript_map)
        adata_usage = compute_isoform_usage(adata, gene_map, min_isoforms=2)

        assert adata_usage.shape[0] == adata.shape[0]  # same cells
        assert adata_usage.shape[1] > 0  # some transcripts kept

        # Check proportions per gene per cell sum to ~1
        X = adata_usage.X
        if scipy.sparse.issparse(X):
            X = X.toarray()

        gene_ids = adata_usage.var['gene_id'].values
        for gene in set(gene_ids):
            gene_mask = gene_ids == gene
            gene_block = X[:, gene_mask]
            row_sums = gene_block.sum(axis=1)
            # For cells with expression, sum should be ~1
            active = row_sums > 0
            if active.sum() > 0:
                np.testing.assert_allclose(
                    row_sums[active], 1.0, atol=1e-10)

    def test_min_isoforms_filter(
            self, transcript_matrix_path, gene_transcript_map):
        """Test that min_isoforms filters out single-isoform genes."""
        try:
            from iris.dual_layer_clustering import (
                load_mex_to_anndata, load_gene_transcript_map,
                compute_isoform_usage)
        except ImportError:
            pytest.skip("scanpy not available")

        adata = load_mex_to_anndata(transcript_matrix_path)
        gene_map = load_gene_transcript_map(gene_transcript_map)

        # With min_isoforms=2, all genes should pass (all have 2-3)
        adata_2 = compute_isoform_usage(adata, gene_map, min_isoforms=2)
        # With min_isoforms=4, no genes should pass
        adata_4 = compute_isoform_usage(adata, gene_map, min_isoforms=4)

        assert adata_2.shape[1] > 0
        assert adata_4.shape[1] == 0

    def test_empty_gene_map(self, transcript_matrix_path):
        """Test with empty gene map returns empty usage matrix."""
        try:
            from iris.dual_layer_clustering import (
                load_mex_to_anndata, compute_isoform_usage)
        except ImportError:
            pytest.skip("scanpy not available")

        adata = load_mex_to_anndata(transcript_matrix_path)
        adata_usage = compute_isoform_usage(adata, {}, min_isoforms=2)
        assert adata_usage.shape[1] == 0


class TestComputeDiversityIndex:
    """Test isoform diversity index computation."""

    def test_shannon_diversity(
            self, transcript_matrix_path, gene_transcript_map):
        """Test Shannon entropy computation."""
        try:
            from iris.dual_layer_clustering import (
                load_mex_to_anndata, load_gene_transcript_map,
                compute_isoform_usage, compute_diversity_index)
        except ImportError:
            pytest.skip("scanpy not available")

        adata = load_mex_to_anndata(transcript_matrix_path)
        gene_map = load_gene_transcript_map(gene_transcript_map)
        adata_usage = compute_isoform_usage(adata, gene_map, min_isoforms=2)

        div_df = compute_diversity_index(
            adata_usage, gene_map, metric='shannon')

        assert len(div_df) == adata.shape[0]
        assert 'barcode' in div_df.columns
        assert 'diversity_index' in div_df.columns
        assert 'n_genes_multi_isoform' in div_df.columns
        # Shannon entropy should be non-negative
        assert (div_df['diversity_index'] >= 0).all()

    def test_simpson_diversity(
            self, transcript_matrix_path, gene_transcript_map):
        """Test Simpson diversity computation."""
        try:
            from iris.dual_layer_clustering import (
                load_mex_to_anndata, load_gene_transcript_map,
                compute_isoform_usage, compute_diversity_index)
        except ImportError:
            pytest.skip("scanpy not available")

        adata = load_mex_to_anndata(transcript_matrix_path)
        gene_map = load_gene_transcript_map(gene_transcript_map)
        adata_usage = compute_isoform_usage(adata, gene_map, min_isoforms=2)

        div_df = compute_diversity_index(
            adata_usage, gene_map, metric='simpson')

        # Simpson index: 1 - sum(p^2), range [0, 1)
        assert (div_df['diversity_index'] >= 0).all()
        assert (div_df['diversity_index'] <= 1).all()

    def test_uniform_usage_has_max_diversity(self):
        """Test that uniform isoform usage gives maximum diversity."""
        import anndata
        from iris.dual_layer_clustering import (
            compute_diversity_index)

        # 1 cell, 1 gene with 4 isoforms, equal usage (0.25 each)
        X = np.array([[0.25, 0.25, 0.25, 0.25]])
        var = pd.DataFrame(
            {'gene_id': ['G1', 'G1', 'G1', 'G1']},
            index=['T1', 'T2', 'T3', 'T4'])
        obs = pd.DataFrame(index=['CELL1'])
        adata_usage = anndata.AnnData(X=X, obs=obs, var=var)

        gene_map = {'T1': 'G1', 'T2': 'G1', 'T3': 'G1', 'T4': 'G1'}
        div_df = compute_diversity_index(
            adata_usage, gene_map, metric='shannon')

        # Shannon entropy of uniform(4) = log2(4) = 2.0
        expected = np.log2(4)
        np.testing.assert_allclose(
            div_df['diversity_index'].values[0], expected, atol=1e-10)


class TestCompareClusterings:
    """Test clustering comparison metrics."""

    def test_identical_clusterings(self):
        """Identical clusterings should give ARI=1, NMI=1."""
        from iris.dual_layer_clustering import (
            compare_clusterings)

        labels = pd.Series(
            ['0', '0', '1', '1', '2', '2'],
            index=[f"C{i}" for i in range(6)])
        result = compare_clusterings(labels, labels)

        assert result['ari'] == pytest.approx(1.0)
        assert result['nmi'] == pytest.approx(1.0)
        assert result['n_common_cells'] == 6
        assert len(result['isoform_specific_clusters']) == 0

    def test_random_clusterings(self):
        """Random clusterings should give low ARI."""
        from iris.dual_layer_clustering import (
            compare_clusterings)

        np.random.seed(42)
        n = 100
        barcodes = [f"C{i}" for i in range(n)]
        gene_cl = pd.Series(
            np.random.choice(['0', '1', '2'], n), index=barcodes)
        iso_cl = pd.Series(
            np.random.choice(['A', 'B', 'C'], n), index=barcodes)

        result = compare_clusterings(gene_cl, iso_cl)
        # ARI near 0 for random
        assert abs(result['ari']) < 0.3
        assert result['n_common_cells'] == n

    def test_isoform_specific_detection(self):
        """Test detection of isoform-specific clusters."""
        from iris.dual_layer_clustering import (
            compare_clusterings)

        barcodes = [f"C{i}" for i in range(20)]

        # Gene clusters: 0 and 1 evenly split
        # Isoform clusters: A and B each draw evenly from both gene clusters
        gene_cl = pd.Series(
            ['0'] * 10 + ['1'] * 10, index=barcodes)
        iso_cl = pd.Series(
            ['A'] * 5 + ['B'] * 5 + ['A'] * 5 + ['B'] * 5,
            index=barcodes)
        # Cluster A: 5 from gene-0, 5 from gene-1 (max_frac=0.5)
        # Cluster B: 5 from gene-0, 5 from gene-1 (max_frac=0.5)

        result = compare_clusterings(gene_cl, iso_cl)
        assert len(result['isoform_specific_clusters']) == 2

    def test_no_common_barcodes(self):
        """Test with disjoint barcode sets."""
        from iris.dual_layer_clustering import (
            compare_clusterings)

        gene_cl = pd.Series(['0', '1'], index=['A', 'B'])
        iso_cl = pd.Series(['0', '1'], index=['C', 'D'])
        result = compare_clusterings(gene_cl, iso_cl)
        assert result['ari'] is None
        assert result['n_common_cells'] == 0


class TestRunClustering:
    """Test the Scanpy clustering pipelines."""

    def test_gene_clustering(self, gene_matrix_dir):
        """Test standard gene-expression clustering runs end-to-end."""
        try:
            from iris.dual_layer_clustering import (
                load_mex_to_anndata, run_clustering)
        except ImportError:
            pytest.skip("scanpy not available")

        adata = load_mex_to_anndata(gene_matrix_dir)
        adata = run_clustering(
            adata, method='louvain', resolution=1.0,
            n_neighbors=10, n_pcs=20)

        assert 'cluster' in adata.obs.columns
        assert 'X_umap' in adata.obsm
        assert adata.obs['cluster'].nunique() >= 1

    def test_isoform_clustering(
            self, transcript_matrix_path, gene_transcript_map):
        """Test isoform-usage clustering runs end-to-end."""
        try:
            from iris.dual_layer_clustering import (
                load_mex_to_anndata, load_gene_transcript_map,
                compute_isoform_usage, run_isoform_clustering)
        except ImportError:
            pytest.skip("scanpy not available")

        adata = load_mex_to_anndata(transcript_matrix_path)
        gene_map = load_gene_transcript_map(gene_transcript_map)
        adata_usage = compute_isoform_usage(adata, gene_map, min_isoforms=2)
        adata_clust = run_isoform_clustering(
            adata_usage, method='louvain', resolution=1.0,
            n_neighbors=10, n_pcs=10)

        assert 'cluster' in adata_clust.obs.columns
        assert 'X_umap' in adata_clust.obsm

    def test_isoform_clustering_empty_matrix(self):
        """Test isoform clustering with empty matrix returns default."""
        try:
            import anndata
            from iris.dual_layer_clustering import (
                run_isoform_clustering)
        except ImportError:
            pytest.skip("scanpy/anndata not available")

        empty_X = scipy.sparse.csr_matrix((10, 0))
        obs = pd.DataFrame(index=[f"C{i}" for i in range(10)])
        adata = anndata.AnnData(X=empty_X, obs=obs)

        result = run_isoform_clustering(
            adata, method='louvain', resolution=1.0,
            n_neighbors=5, n_pcs=5)
        assert (result.obs['cluster'] == '0').all()


class TestJointEmbedding:
    """Test joint gene+isoform embedding."""

    def test_joint_embedding_end_to_end(
            self, gene_matrix_dir, transcript_matrix_path,
            gene_transcript_map):
        """Test joint embedding produces valid output."""
        try:
            from iris.dual_layer_clustering import (
                load_mex_to_anndata, load_gene_transcript_map,
                compute_isoform_usage, run_clustering,
                run_isoform_clustering, compute_joint_embedding)
        except ImportError:
            pytest.skip("scanpy not available")

        adata_gene = load_mex_to_anndata(gene_matrix_dir)
        adata_tx = load_mex_to_anndata(transcript_matrix_path)
        gene_map = load_gene_transcript_map(gene_transcript_map)

        adata_usage = compute_isoform_usage(adata_tx, gene_map)
        adata_gene = run_clustering(
            adata_gene, 'leiden', 1.0, 10, 10)
        adata_iso = run_isoform_clustering(
            adata_usage, 'leiden', 1.0, 10, 10)

        adata_joint = compute_joint_embedding(
            adata_gene, adata_iso, n_neighbors=10, n_pcs=10,
            method='louvain', resolution=1.0)

        assert adata_joint.shape[0] > 0
        assert 'cluster' in adata_joint.obs.columns
        assert 'X_umap' in adata_joint.obsm
        assert adata_joint.obsm['X_umap'].shape[1] == 2
