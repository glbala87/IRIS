"""Tests for novel_isoform_discovery module (isosceles Module 6)."""
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
def annotated_gtf(tmp_path):
    """Create a mock gffcompare annotated GTF file.

    Contains known (=) and novel (j, u, o) transcripts.
    """
    gtf_file = tmp_path / "gffcompare.annotated.gtf"
    lines = [
        '# gffcompare annotated output',
        'chr1\tStringTie\ttranscript\t1000\t5000\t.\t+\t.\t'
        'transcript_id "KNOWN.1"; gene_id "GENE_A"; '
        'class_code "="; ref_gene_id "GENE_A"; cmp_ref "TX_REF1";',

        'chr1\tStringTie\ttranscript\t1000\t6000\t.\t+\t.\t'
        'transcript_id "NOVEL_J.1"; gene_id "GENE_A"; '
        'class_code "j"; ref_gene_id "GENE_A"; cmp_ref "TX_REF1";',

        'chr1\tStringTie\ttranscript\t8000\t9000\t.\t+\t.\t'
        'transcript_id "NOVEL_U.1"; gene_id "NOVEL_U.1"; '
        'class_code "u";',

        'chr2\tStringTie\ttranscript\t2000\t4000\t.\t+\t.\t'
        'transcript_id "NOVEL_O.1"; gene_id "GENE_B"; '
        'class_code "o"; ref_gene_id "GENE_B"; cmp_ref "TX_REF2";',

        'chr2\tStringTie\ttranscript\t2000\t4000\t.\t+\t.\t'
        'transcript_id "KNOWN.2"; gene_id "GENE_B"; '
        'class_code "="; ref_gene_id "GENE_B"; cmp_ref "TX_REF2";',

        'chr2\tStringTie\ttranscript\t5000\t7000\t.\t-\t.\t'
        'transcript_id "NOVEL_X.1"; gene_id "GENE_C"; '
        'class_code "x"; ref_gene_id "GENE_C"; cmp_ref "TX_REF3";',
    ]
    gtf_file.write_text('\n'.join(lines) + '\n')
    return gtf_file


@pytest.fixture
def transcript_matrix_dir(tmp_path):
    """Create a transcript matrix with known + novel transcripts.

    NOVEL_J.1 is enriched in cluster 0.
    NOVEL_O.1 is enriched in cluster 1.
    NOVEL_X.1 is expressed broadly.
    """
    out = tmp_path / "tx_matrix"
    out.mkdir()
    n_cells = 60
    np.random.seed(42)

    tx_names = [
        'KNOWN.1', 'KNOWN.2',
        'NOVEL_J.1', 'NOVEL_O.1', 'NOVEL_U.1', 'NOVEL_X.1'
    ]
    n_tx = len(tx_names)
    data = np.zeros((n_tx, n_cells), dtype=np.float32)

    # Known transcripts: expressed broadly
    data[0, :] = np.random.poisson(5, n_cells)
    data[1, :] = np.random.poisson(3, n_cells)

    # NOVEL_J.1: enriched in cluster 0 (first 30 cells)
    data[2, :30] = np.random.poisson(8, 30)
    data[2, 30:] = np.random.poisson(1, 30)

    # NOVEL_O.1: enriched in cluster 1 (last 30 cells)
    data[3, :30] = np.random.poisson(1, 30)
    data[3, 30:] = np.random.poisson(8, 30)

    # NOVEL_U.1: very sparse (should be filtered)
    data[4, 0] = 1

    # NOVEL_X.1: expressed broadly (not cluster-specific)
    data[5, :] = np.random.poisson(4, n_cells)

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
def clusters_tsv(tmp_path, transcript_matrix_dir):
    """Cluster assignments: first 30 = cluster 0, rest = cluster 1."""
    _, barcodes = transcript_matrix_dir
    f = tmp_path / "clusters.tsv"
    with open(f, 'w') as fh:
        fh.write("barcode\tcluster\n")
        for i, bc in enumerate(barcodes):
            fh.write(f"{bc}\t{'0' if i < 30 else '1'}\n")
    return f


@pytest.fixture
def gene_transcript_map(tmp_path):
    """Gene-transcript map."""
    f = tmp_path / "map.tsv"
    with open(f, 'w') as fh:
        fh.write("transcript_id\tgene_id\n")
        fh.write("KNOWN.1\tGENE_A\n")
        fh.write("KNOWN.2\tGENE_B\n")
        fh.write("NOVEL_J.1\tGENE_A\n")
        fh.write("NOVEL_O.1\tGENE_B\n")
        fh.write("NOVEL_U.1\tNOVEL_U.1\n")
        fh.write("NOVEL_X.1\tGENE_C\n")
    return f


class TestParseGffcompareGtf:
    """Test GTF parsing."""

    def test_parse_annotated_gtf(self, annotated_gtf):
        """Test parsing a gffcompare annotated GTF."""
        from iris.novel_isoform_discovery import (
            parse_gffcompare_gtf)

        df = parse_gffcompare_gtf(annotated_gtf)
        assert len(df) == 6
        assert 'transcript_id' in df.columns
        assert 'class_code' in df.columns

        known = df[df['class_code'] == '=']
        assert len(known) == 2

        novel_j = df[df['class_code'] == 'j']
        assert len(novel_j) == 1
        assert novel_j.iloc[0]['transcript_id'] == 'NOVEL_J.1'
        assert novel_j.iloc[0]['ref_gene_id'] == 'GENE_A'

    def test_parse_multiple_gtfs(self, tmp_path):
        """Test parsing multiple GTF files."""
        from iris.novel_isoform_discovery import (
            parse_multiple_gtfs)

        # Create two small GTFs
        gtf1 = tmp_path / "chr1.gtf"
        gtf1.write_text(
            'chr1\tST\ttranscript\t1\t100\t.\t+\t.\t'
            'transcript_id "TX1"; gene_id "G1"; class_code "j";\n')
        gtf2 = tmp_path / "chr2.gtf"
        gtf2.write_text(
            'chr2\tST\ttranscript\t1\t100\t.\t+\t.\t'
            'transcript_id "TX2"; gene_id "G2"; class_code "=";\n')

        df = parse_multiple_gtfs([gtf1, gtf2])
        assert len(df) == 2


class TestClassifyTranscripts:
    """Test transcript classification."""

    def test_classify(self, annotated_gtf):
        """Test novelty classification based on class codes."""
        from iris.novel_isoform_discovery import (
            parse_gffcompare_gtf, classify_transcripts)

        df = parse_gffcompare_gtf(annotated_gtf)
        df = classify_transcripts(df)

        assert 'novelty' in df.columns
        known = df[df['novelty'] == 'known']
        novel = df[df['novelty'] == 'novel']
        assert len(known) == 2   # '=' codes
        assert len(novel) == 4   # 'j', 'u', 'o', 'x' codes


class TestBuildNovelCatalog:
    """Test novel isoform catalog building."""

    def test_build_catalog(self, annotated_gtf, transcript_matrix_dir):
        """Test catalog building with expression filtering."""
        try:
            import scanpy as sc
        except ImportError:
            pytest.skip("scanpy not available")

        from iris.novel_isoform_discovery import (
            parse_gffcompare_gtf, classify_transcripts,
            build_novel_catalog)

        matrix_path, _ = transcript_matrix_dir
        adata = sc.read_10x_mtx(str(matrix_path), var_names='gene_symbols')

        annotation = classify_transcripts(
            parse_gffcompare_gtf(annotated_gtf))
        catalog = build_novel_catalog(
            annotation, adata, min_cells=3, min_counts=5)

        assert len(catalog) > 0
        # NOVEL_U.1 should be filtered (only 1 cell, 1 count)
        assert 'NOVEL_U.1' not in catalog['transcript_id'].values
        # NOVEL_J.1 and NOVEL_O.1 should be present
        assert 'NOVEL_J.1' in catalog['transcript_id'].values
        assert 'NOVEL_O.1' in catalog['transcript_id'].values

        # Check required columns
        for col in ['n_cells', 'total_counts', 'mean_expr_per_cell']:
            assert col in catalog.columns

    def test_catalog_with_no_novel(self, transcript_matrix_dir):
        """Test with annotations containing no novel transcripts."""
        try:
            import scanpy as sc
        except ImportError:
            pytest.skip("scanpy not available")

        from iris.novel_isoform_discovery import (
            classify_transcripts, build_novel_catalog)

        matrix_path, _ = transcript_matrix_dir
        adata = sc.read_10x_mtx(str(matrix_path), var_names='gene_symbols')

        # All known
        annotation = pd.DataFrame({
            'transcript_id': ['TX1'],
            'class_code': ['='],
            'novelty': ['known'],
            'gene_id': ['G1'],
            'ref_gene_id': ['G1'],
            'ref_transcript_id': ['REF1'],
            'chrom': ['chr1'],
        })
        catalog = build_novel_catalog(annotation, adata)
        assert len(catalog) == 0


class TestClusterEnrichment:
    """Test cluster enrichment analysis."""

    def test_enrichment_detects_signal(
            self, annotated_gtf, transcript_matrix_dir, clusters_tsv):
        """Test Fisher's exact test detects planted enrichment."""
        try:
            import scanpy as sc
        except ImportError:
            pytest.skip("scanpy not available")

        from iris.novel_isoform_discovery import (
            parse_gffcompare_gtf, classify_transcripts,
            build_novel_catalog, test_cluster_enrichment)

        matrix_path, _ = transcript_matrix_dir
        adata = sc.read_10x_mtx(str(matrix_path), var_names='gene_symbols')

        annotation = classify_transcripts(
            parse_gffcompare_gtf(annotated_gtf))
        catalog = build_novel_catalog(
            annotation, adata, min_cells=3, min_counts=5)

        clusters_df = pd.read_csv(clusters_tsv, sep='\t')
        cluster_labels = pd.Series(
            clusters_df['cluster'].astype(str).values,
            index=clusters_df['barcode'].values)

        enrichment = test_cluster_enrichment(
            adata, catalog, cluster_labels, fdr_threshold=0.05)

        assert len(enrichment) > 0
        assert 'pvalue_adj' in enrichment.columns
        assert 'fold_enrichment' in enrichment.columns

        # NOVEL_J.1 should be enriched in cluster 0
        j1_cl0 = enrichment[
            (enrichment['transcript_id'] == 'NOVEL_J.1') &
            (enrichment['cluster'] == '0')]
        if len(j1_cl0) > 0:
            assert j1_cl0.iloc[0]['fold_enrichment'] > 1.0

        # NOVEL_O.1 should be enriched in cluster 1
        o1_cl1 = enrichment[
            (enrichment['transcript_id'] == 'NOVEL_O.1') &
            (enrichment['cluster'] == '1')]
        if len(o1_cl1) > 0:
            assert o1_cl1.iloc[0]['fold_enrichment'] > 1.0


class TestSpecificityScore:
    """Test cluster specificity scoring."""

    def test_specificity_high_for_specific(self):
        """Isoform expressed in one cluster gets high specificity."""
        from iris.novel_isoform_discovery import (
            compute_specificity_score)

        enrichment = pd.DataFrame([
            {'transcript_id': 'TX1', 'gene_id': 'G1',
             'class_code': 'j', 'ref_gene_id': 'G1',
             'cluster': '0', 'n_expressing_in_cluster': 20,
             'n_expressing_total': 22, 'n_cells_in_cluster': 30,
             'pct_in_cluster': 0.67, 'pct_in_rest': 0.07,
             'fold_enrichment': 3.5, 'pvalue': 1e-10,
             'pvalue_adj': 1e-8},
            {'transcript_id': 'TX1', 'gene_id': 'G1',
             'class_code': 'j', 'ref_gene_id': 'G1',
             'cluster': '1', 'n_expressing_in_cluster': 2,
             'n_expressing_total': 22, 'n_cells_in_cluster': 30,
             'pct_in_cluster': 0.07, 'pct_in_rest': 0.67,
             'fold_enrichment': 0.3, 'pvalue': 0.99,
             'pvalue_adj': 0.99},
        ])

        spec = compute_specificity_score(enrichment, fdr_threshold=0.05)
        assert len(spec) == 1
        assert spec.iloc[0]['specificity_score'] > 1.0
        assert spec.iloc[0]['best_cluster'] == '0'
        assert spec.iloc[0]['n_significant_clusters'] == 1

    def test_specificity_low_for_broad(self):
        """Isoform expressed in all clusters gets low specificity."""
        from iris.novel_isoform_discovery import (
            compute_specificity_score)

        enrichment = pd.DataFrame([
            {'transcript_id': 'TX1', 'gene_id': 'G1',
             'class_code': 'j', 'ref_gene_id': 'G1',
             'cluster': '0', 'n_expressing_in_cluster': 10,
             'n_expressing_total': 20, 'n_cells_in_cluster': 30,
             'pct_in_cluster': 0.33, 'pct_in_rest': 0.33,
             'fold_enrichment': 1.0, 'pvalue': 0.5,
             'pvalue_adj': 0.5},
            {'transcript_id': 'TX1', 'gene_id': 'G1',
             'class_code': 'j', 'ref_gene_id': 'G1',
             'cluster': '1', 'n_expressing_in_cluster': 10,
             'n_expressing_total': 20, 'n_cells_in_cluster': 30,
             'pct_in_cluster': 0.33, 'pct_in_rest': 0.33,
             'fold_enrichment': 1.0, 'pvalue': 0.5,
             'pvalue_adj': 0.5},
        ])

        spec = compute_specificity_score(enrichment, fdr_threshold=0.05)
        assert len(spec) == 1
        # Uniform distribution → entropy = 1 → specificity ≈ 0
        assert spec.iloc[0]['specificity_score'] < 0.1
        assert spec.iloc[0]['n_significant_clusters'] == 0


class TestBHCorrection:
    """Test BH correction."""

    def test_correction(self):
        """Basic BH correction test."""
        from iris.novel_isoform_discovery import _bh_correct

        pvals = np.array([0.01, 0.04, 0.03, 0.10])
        adj = _bh_correct(pvals)
        assert (adj >= pvals).all()
        assert (adj <= 1.0).all()

    def test_nan_handling(self):
        """NaN p-values are preserved."""
        from iris.novel_isoform_discovery import _bh_correct

        pvals = np.array([0.01, np.nan, 0.05])
        adj = _bh_correct(pvals)
        assert np.isnan(adj[1])
        assert not np.isnan(adj[0])


class TestEndToEnd:
    """End-to-end test of the novel isoform discovery pipeline."""

    def test_full_pipeline(
            self, annotated_gtf, transcript_matrix_dir,
            clusters_tsv, gene_transcript_map, tmp_path):
        """Test full pipeline produces valid outputs."""
        try:
            import scanpy  # noqa: F401
        except ImportError:
            pytest.skip("scanpy not available")

        from iris.novel_isoform_discovery import main

        matrix_path, _ = transcript_matrix_dir

        class Args:
            pass

        args = Args()
        args.transcript_matrix_dir = matrix_path
        args.annotated_gtfs = [annotated_gtf]
        args.clusters = clusters_tsv
        args.gene_transcript_map = gene_transcript_map
        args.output_novel_catalog = tmp_path / "catalog.tsv"
        args.output_cluster_enrichment = tmp_path / "enrichment.tsv"
        args.output_summary = tmp_path / "summary.json"
        args.min_cells = 3
        args.min_counts = 5
        args.enrichment_fdr = 0.05
        args.cluster_column = "cluster"

        main(args)

        # Check outputs
        assert args.output_novel_catalog.exists()
        assert args.output_cluster_enrichment.exists()
        assert args.output_summary.exists()

        catalog = pd.read_csv(args.output_novel_catalog, sep='\t')
        assert len(catalog) > 0

        enrichment = pd.read_csv(
            args.output_cluster_enrichment, sep='\t')
        assert len(enrichment) > 0

        with open(args.output_summary) as f:
            summary = json.load(f)
        assert summary['n_novel_in_matrix'] > 0
        assert 'class_code_distribution' in summary
