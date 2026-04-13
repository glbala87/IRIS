"""Tests for allele_specific_expression module (isosceles Module 9)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestAllelicImbalance:
    """Test binomial test for allelic imbalance."""

    def test_balanced_counts(self):
        """Balanced allele counts should not be significant."""
        from iris.allele_specific_expression import (
            compute_allelic_imbalance)

        pval, ratio = compute_allelic_imbalance(50, 50)
        assert pval > 0.05
        assert ratio == pytest.approx(0.5)

    def test_imbalanced_counts(self):
        """Strongly imbalanced counts should be significant."""
        from iris.allele_specific_expression import (
            compute_allelic_imbalance)

        pval, ratio = compute_allelic_imbalance(95, 5)
        assert pval < 0.05
        assert ratio == pytest.approx(0.05)

    def test_zero_counts(self):
        """Zero total counts should return p=1 and NaN ratio."""
        from iris.allele_specific_expression import (
            compute_allelic_imbalance)

        pval, ratio = compute_allelic_imbalance(0, 0)
        assert pval == 1.0
        assert np.isnan(ratio)


class TestBHCorrect:
    """Test Benjamini-Hochberg correction."""

    def test_basic_correction(self):
        """Adjusted p-values should be >= original and <= 1."""
        from iris.allele_specific_expression import _bh_correct

        pvals = np.array([0.01, 0.04, 0.03, 0.10])
        adj = _bh_correct(pvals)
        assert (adj >= pvals).all()
        assert (adj <= 1.0).all()

    def test_nan_handling(self):
        """NaN p-values should be preserved."""
        from iris.allele_specific_expression import _bh_correct

        pvals = np.array([0.01, np.nan, 0.05])
        adj = _bh_correct(pvals)
        assert np.isnan(adj[1])
        assert not np.isnan(adj[0])
        assert not np.isnan(adj[2])

    def test_all_nan(self):
        """All-NaN input should return all NaN."""
        from iris.allele_specific_expression import _bh_correct

        pvals = np.array([np.nan, np.nan])
        adj = _bh_correct(pvals)
        assert np.isnan(adj).all()

    def test_empty(self):
        """Empty input should return empty output."""
        from iris.allele_specific_expression import _bh_correct

        adj = _bh_correct(np.array([]))
        assert len(adj) == 0


class TestLoadVariantsFromVcf:
    """Test VCF parsing."""

    def test_parse_vcf(self, tmp_path):
        """Test parsing a simple VCF with het SNVs."""
        from iris.allele_specific_expression import (
            load_variants_from_vcf)

        vcf = tmp_path / "test.vcf"
        lines = [
            "##fileformat=VCFv4.2",
            "##FORMAT=<ID=GT,Number=1,Type=String>",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE",
            "chr1\t1000\t.\tA\tG\t50\tPASS\t.\tGT\t0/1",
            "chr1\t2000\t.\tC\tT\t50\tPASS\t.\tGT\t1/1",   # hom, skip
            "chr2\t3000\t.\tG\tA\t50\tPASS\t.\tGT\t0|1",
            "chr2\t4000\t.\tAA\tG\t50\tPASS\t.\tGT\t0/1",  # indel, skip
        ]
        vcf.write_text('\n'.join(lines) + '\n')

        variants = load_variants_from_vcf(vcf)
        assert len(variants) == 2
        assert variants.iloc[0]['chrom'] == 'chr1'
        assert variants.iloc[0]['pos'] == 999  # 0-based
        assert variants.iloc[0]['ref'] == 'A'
        assert variants.iloc[0]['alt'] == 'G'
        assert variants.iloc[1]['chrom'] == 'chr2'

    def test_empty_vcf(self, tmp_path):
        """Test parsing a VCF with no het variants."""
        from iris.allele_specific_expression import (
            load_variants_from_vcf)

        vcf = tmp_path / "empty.vcf"
        lines = [
            "##fileformat=VCFv4.2",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE",
            "chr1\t100\t.\tA\tG\t50\tPASS\t.\tGT\t1/1",
        ]
        vcf.write_text('\n'.join(lines) + '\n')

        variants = load_variants_from_vcf(vcf)
        assert len(variants) == 0


class TestExtractAndCountFromBam:
    """Tests requiring BAM files (skipped without real alignment data)."""

    def test_extract_variants_from_bam(self):
        """Skip: requires indexed BAM with pileup data."""
        pytest.skip("requires BAM")

    def test_count_alleles_per_cell(self):
        """Skip: requires indexed BAM with CB tags."""
        pytest.skip("requires BAM")


class TestAggregateByCluster:
    """Test cluster-level aggregation and Fisher's exact test."""

    def test_aggregate(self):
        """Test aggregation and imbalance detection per cluster."""
        from iris.allele_specific_expression import (
            aggregate_by_cluster)

        # Synthetic allele counts: 2 variants, 2 clusters
        allele_counts = pd.DataFrame([
            # Variant 1, cluster A cells: strongly imbalanced
            {'chrom': 'chr1', 'pos': 100, 'barcode': 'C1',
             'ref_count': 20, 'alt_count': 2},
            {'chrom': 'chr1', 'pos': 100, 'barcode': 'C2',
             'ref_count': 18, 'alt_count': 3},
            # Variant 1, cluster B cells: opposite imbalance
            {'chrom': 'chr1', 'pos': 100, 'barcode': 'C3',
             'ref_count': 2, 'alt_count': 19},
            {'chrom': 'chr1', 'pos': 100, 'barcode': 'C4',
             'ref_count': 3, 'alt_count': 22},
            # Variant 2, cluster A: balanced
            {'chrom': 'chr2', 'pos': 200, 'barcode': 'C1',
             'ref_count': 10, 'alt_count': 12},
            {'chrom': 'chr2', 'pos': 200, 'barcode': 'C2',
             'ref_count': 11, 'alt_count': 9},
            # Variant 2, cluster B: balanced
            {'chrom': 'chr2', 'pos': 200, 'barcode': 'C3',
             'ref_count': 10, 'alt_count': 11},
            {'chrom': 'chr2', 'pos': 200, 'barcode': 'C4',
             'ref_count': 12, 'alt_count': 10},
        ])

        cluster_labels = pd.Series(
            {'C1': 'A', 'C2': 'A', 'C3': 'B', 'C4': 'B'})

        cluster_ase, diff_ase = aggregate_by_cluster(
            allele_counts, cluster_labels)

        # Should have 4 rows: 2 variants x 2 clusters
        assert len(cluster_ase) == 4
        assert 'pvalue' in cluster_ase.columns
        assert 'pvalue_adj' in cluster_ase.columns

        # Variant 1 should show imbalance in both clusters
        v1 = cluster_ase[cluster_ase['pos'] == 100]
        assert (v1['pvalue'] < 0.05).all()

        # Differential ASE: variant 1 should differ between clusters
        assert len(diff_ase) > 0
        assert 'fisher_pvalue' in diff_ase.columns
        v1_diff = diff_ase[diff_ase['pos'] == 100]
        assert len(v1_diff) == 1
        assert v1_diff.iloc[0]['fisher_pvalue'] < 0.05

    def test_empty_input(self):
        """Test with empty allele counts."""
        from iris.allele_specific_expression import (
            aggregate_by_cluster)

        empty = pd.DataFrame(
            columns=['chrom', 'pos', 'barcode', 'ref_count', 'alt_count'])
        cluster_labels = pd.Series({'C1': 'A'})

        cluster_ase, diff_ase = aggregate_by_cluster(empty, cluster_labels)
        assert len(cluster_ase) == 0
        assert len(diff_ase) == 0
