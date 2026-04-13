"""Tests for IRIS multi-sample comparison module."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _create_sample_dir(base_path, sample_name, seed=42):
    """Create a mock IRIS output directory for one sample."""
    np.random.seed(seed)
    d = base_path / sample_name
    d.mkdir(parents=True)

    # DTU results
    dtu_df = pd.DataFrame({
        'gene': ['GENE_A', 'GENE_B', 'GENE_C', 'GENE_D', 'GENE_E'],
        'cluster_a': ['0'] * 5,
        'cluster_b': ['1'] * 5,
        'test_statistic': [25.0, 8.0, 30.0, 2.0, 15.0],
        'pvalue': [1e-6, 0.01, 1e-8, 0.5, 0.001],
        'pvalue_adj': [5e-6, 0.03, 5e-8, 0.6, 0.004],
        'effect_size': [0.8, 0.4, 0.9, 0.1, 0.6],
        'n_transcripts': [3, 2, 4, 2, 3],
    })
    dtu_df.to_csv(d / "dtu_results.tsv", sep='\t', index=False)

    # Switching
    sw_df = pd.DataFrame({
        'gene': ['GENE_A', 'GENE_C'],
        'cluster_a': ['0', '0'],
        'cluster_b': ['1', '1'],
        'dominant_transcript_a': ['TX_A.0', 'TX_C.0'],
        'proportion_a': [0.8, 0.75],
        'dominant_transcript_b': ['TX_A.1', 'TX_C.2'],
        'proportion_b': [0.7, 0.8],
        'switching_score': [0.6, 0.55],
    })
    sw_df.to_csv(d / "isoform_switching.tsv", sep='\t', index=False)

    # Novel catalog
    novel_df = pd.DataFrame({
        'transcript_id': ['NOVEL_1', 'NOVEL_2', 'NOVEL_3'],
        'gene_id': ['G1', 'G2', 'G3'],
        'class_code': ['j', 'u', 'o'],
        'n_cells': [20, 10, 5],
        'total_counts': [100, 40, 15],
    })
    novel_df.to_csv(
        d / "novel_isoform_catalog.tsv", sep='\t', index=False)

    # Cell type annotations (per-cell, matching what load_sample_results expects)
    ct_df = pd.DataFrame({
        'barcode': [f"CELL{i:04d}" for i in range(70)],
        'cluster': [str(i % 3) for i in range(70)],
        'cell_type': ['T cells'] * 30 + ['B cells'] * 25 +
                     ['NK cells'] * 15,
    })
    ct_df.to_csv(
        d / "cell_type_annotations.tsv", sep='\t', index=False)

    return d


@pytest.fixture
def two_sample_dirs(tmp_path):
    """Create two sample output directories."""
    s1 = _create_sample_dir(tmp_path, "sample1", seed=42)
    s2 = _create_sample_dir(tmp_path, "sample2", seed=99)
    return [s1, s2]


class TestLoadSampleResults:
    """Test sample loading."""

    def test_load_complete(self, two_sample_dirs):
        from iris.multi_sample import load_sample_results
        result = load_sample_results(two_sample_dirs[0], "sample1")
        assert result['name'] == 'sample1'
        assert 'dtu' in result
        assert 'switching' in result
        assert 'novel_catalog' in result

    def test_load_empty_dir(self, tmp_path):
        from iris.multi_sample import load_sample_results
        empty = tmp_path / "empty"
        empty.mkdir()
        result = load_sample_results(empty, "empty")
        assert result['name'] == 'empty'
        assert result.get('dtu') is None


class TestCompareDTU:
    """Test cross-sample DTU comparison."""

    def test_compare_dtu(self, two_sample_dirs):
        from iris.multi_sample import (
            load_sample_results, compare_dtu_across_samples)

        samples = [
            load_sample_results(d, f"s{i}")
            for i, d in enumerate(two_sample_dirs)]
        result = compare_dtu_across_samples(samples, fdr_threshold=0.05)

        assert len(result) > 0
        assert 'gene' in result.columns
        assert 'n_samples_significant' in result.columns
        assert 'conserved' in result.columns

        # GENE_A and GENE_C should be conserved (significant in both)
        conserved = result[result['conserved']]
        assert 'GENE_A' in conserved['gene'].values
        assert 'GENE_C' in conserved['gene'].values

    def test_no_dtu_data(self, tmp_path):
        from iris.multi_sample import (
            load_sample_results, compare_dtu_across_samples)
        empty = tmp_path / "e1"
        empty.mkdir()
        samples = [load_sample_results(empty, "e1")]
        result = compare_dtu_across_samples(samples)
        assert len(result) == 0


class TestCompareNovel:
    """Test cross-sample novel isoform comparison."""

    def test_compare_novel(self, two_sample_dirs):
        from iris.multi_sample import (
            load_sample_results, compare_novel_isoforms)

        samples = [
            load_sample_results(d, f"s{i}")
            for i, d in enumerate(two_sample_dirs)]
        result = compare_novel_isoforms(samples)

        assert len(result) > 0
        assert 'transcript_id' in result.columns
        assert 'n_samples' in result.columns

        # NOVEL_1, NOVEL_2, NOVEL_3 appear in both samples
        shared = result[result['n_samples'] == 2]
        assert len(shared) == 3


class TestCompareSwitching:
    """Test cross-sample switching comparison."""

    def test_conserved_switches(self, two_sample_dirs):
        from iris.multi_sample import (
            load_sample_results, compare_switching_events)

        samples = [
            load_sample_results(d, f"s{i}")
            for i, d in enumerate(two_sample_dirs)]
        result = compare_switching_events(samples)

        assert len(result) > 0
        # GENE_A switches TX_A.0->TX_A.1 in both samples
        conserved = result[result['n_samples'] == 2]
        assert 'GENE_A' in conserved['gene'].values


class TestCompareCellTypes:
    """Test cell type composition comparison."""

    def test_composition(self, two_sample_dirs):
        from iris.multi_sample import (
            load_sample_results, compare_cell_type_composition)

        samples = [
            load_sample_results(d, f"s{i}")
            for i, d in enumerate(two_sample_dirs)]
        result = compare_cell_type_composition(samples)

        assert isinstance(result, tuple)
        comp_df, fisher_df = result
        assert len(comp_df) > 0
        assert 'sample' in comp_df.columns
        assert len(fisher_df) > 0
        assert 'cell_type' in fisher_df.columns


class TestEndToEnd:
    """End-to-end multi-sample comparison."""

    def test_main(self, two_sample_dirs, tmp_path):
        from iris.multi_sample import main

        class Args:
            pass
        args = Args()
        args.sample_dirs = two_sample_dirs
        args.sample_names = ['sample1', 'sample2']
        args.output_dir = tmp_path / "comparison"
        args.fdr_threshold = 0.05

        main(args)

        out = tmp_path / "comparison"
        assert out.exists()
        assert (out / "comparison_summary.json").exists()

        with open(out / "comparison_summary.json") as f:
            summary = json.load(f)
        assert summary['n_samples'] == 2
