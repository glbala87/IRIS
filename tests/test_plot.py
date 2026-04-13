"""Tests for IRIS plot module."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def iris_output_dir(tmp_path):
    """Create a mock IRIS output directory with sample data."""
    out = tmp_path / "iris_results"
    out.mkdir()

    n_cells = 40

    # joint UMAP
    umap_df = pd.DataFrame({
        'barcode': [f"CELL{i:04d}" for i in range(n_cells)],
        'D1': np.random.randn(n_cells),
        'D2': np.random.randn(n_cells),
        'cluster': [str(i % 3) for i in range(n_cells)],
    })
    umap_df.to_csv(out / "joint.umap.tsv", sep='\t', index=False)

    # Cell type annotations
    annot_df = pd.DataFrame({
        'barcode': [f"CELL{i:04d}" for i in range(n_cells)],
        'cluster': [str(i % 3) for i in range(n_cells)],
        'cell_type': ['T cells'] * 14 + ['B cells'] * 13 +
                     ['NK cells'] * 13,
    })
    annot_df.to_csv(
        out / "cell_type_annotations.tsv", sep='\t', index=False)

    # Diversity
    div_df = pd.DataFrame({
        'barcode': [f"CELL{i:04d}" for i in range(n_cells)],
        'diversity_index': np.random.uniform(0, 2, n_cells),
        'n_genes_multi_isoform': np.random.randint(5, 50, n_cells),
    })
    div_df.to_csv(
        out / "isoform_diversity.tsv", sep='\t', index=False)

    # DTU results
    dtu_df = pd.DataFrame({
        'gene': [f"GENE_{i}" for i in range(20)],
        'cluster_a': ['0'] * 20,
        'cluster_b': ['1'] * 20,
        'test_statistic': np.random.uniform(0, 50, 20),
        'pvalue': np.concatenate([
            np.random.uniform(1e-10, 0.001, 8),
            np.random.uniform(0.1, 1, 12)]),
        'pvalue_adj': np.concatenate([
            np.random.uniform(1e-8, 0.01, 8),
            np.random.uniform(0.1, 1, 12)]),
        'effect_size': np.random.uniform(0, 1, 20),
        'n_transcripts': np.random.randint(2, 5, 20),
    })
    dtu_df.to_csv(out / "dtu_results.tsv", sep='\t', index=False)

    # Switching events
    sw_df = pd.DataFrame({
        'gene': [f"GENE_{i}" for i in range(5)],
        'cluster_a': ['0'] * 5,
        'cluster_b': ['1'] * 5,
        'dominant_transcript_a': [f"TX_{i}.0" for i in range(5)],
        'proportion_a': [0.8] * 5,
        'dominant_transcript_b': [f"TX_{i}.1" for i in range(5)],
        'proportion_b': [0.75] * 5,
        'switching_score': [0.6] * 5,
    })
    sw_df.to_csv(out / "isoform_switching.tsv", sep='\t', index=False)

    # Novel isoform summary
    novel_summary = {
        'n_novel_in_annotations': 100,
        'n_novel_in_matrix': 30,
        'n_significant_enrichments': 12,
        'n_cluster_specific_isoforms': 5,
        'class_code_distribution': {'j': 15, 'o': 8, 'u': 5, 'x': 2},
    }
    with open(out / "novel_isoform_summary.json", 'w') as f:
        json.dump(novel_summary, f)

    # Cluster comparison
    comp = {
        'ari': 0.45,
        'nmi': 0.62,
        'n_common_cells': 40,
        'isoform_specific_clusters': ['iso_2'],
        'contingency_table': {
            'gene_clusters': ['0', '1', '2'],
            'isoform_clusters': ['A', 'B', 'C'],
            'counts': [[12, 2, 0], [1, 11, 2], [1, 1, 10]],
        },
    }
    with open(out / "cluster_comparison.json", 'w') as f:
        json.dump(comp, f)

    # Pseudotime
    pt_df = pd.DataFrame({
        'barcode': [f"CELL{i:04d}" for i in range(n_cells)],
        'dpt_pseudotime': np.linspace(0, 1, n_cells),
    })
    pt_df.to_csv(out / "pseudotime.tsv", sep='\t', index=False)

    return out


class TestSetStyle:
    """Test style configuration."""

    def test_publication_style(self):
        from iris.plot import set_style
        set_style('publication')
        import matplotlib.pyplot as plt
        assert plt.rcParams['font.size'] <= 10

    def test_presentation_style(self):
        from iris.plot import set_style
        set_style('presentation')
        import matplotlib.pyplot as plt
        assert plt.rcParams['font.size'] >= 14


class TestPlotFunctions:
    """Test individual plot functions."""

    def test_joint_umap(self, iris_output_dir, tmp_path):
        from iris.plot import plot_joint_umap
        save_dir = tmp_path / "figs"
        save_dir.mkdir()
        plot_joint_umap(iris_output_dir, save_dir, 'png', 150)
        assert (save_dir / "umap_clusters.png").exists()

    def test_dtu_volcano(self, iris_output_dir, tmp_path):
        from iris.plot import plot_dtu_volcano
        save_dir = tmp_path / "figs"
        save_dir.mkdir()
        plot_dtu_volcano(iris_output_dir, save_dir, 'png', 150)
        assert (save_dir / "dtu_volcano.png").exists()

    def test_novel_barplot(self, iris_output_dir, tmp_path):
        from iris.plot import plot_novel_isoform_barplot
        save_dir = tmp_path / "figs"
        save_dir.mkdir()
        plot_novel_isoform_barplot(iris_output_dir, save_dir, 'png', 150)
        assert (save_dir / "novel_class_codes.png").exists()

    def test_trajectory_stream(self, iris_output_dir, tmp_path):
        from iris.plot import plot_trajectory_stream
        save_dir = tmp_path / "figs"
        save_dir.mkdir()
        plot_trajectory_stream(iris_output_dir, save_dir, 'png', 150)
        assert (save_dir / "trajectory_pseudotime.png").exists()

    def test_cluster_comparison(self, iris_output_dir, tmp_path):
        from iris.plot import plot_cluster_comparison
        save_dir = tmp_path / "figs"
        save_dir.mkdir()
        plot_cluster_comparison(iris_output_dir, save_dir, 'png', 150)
        assert (save_dir / "cluster_contingency.png").exists()

    def test_missing_data_skips(self, tmp_path):
        """Plot functions should return gracefully when files missing."""
        from iris.plot import plot_joint_umap
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        save_dir = tmp_path / "figs"
        save_dir.mkdir()
        # Should not raise
        plot_joint_umap(empty_dir, save_dir, 'png', 150)
        assert not (save_dir / "umap_clusters.png").exists()


class TestEndToEnd:
    """Test full plotting pipeline."""

    def test_main(self, iris_output_dir, tmp_path):
        from iris.plot import main

        class Args:
            pass
        args = Args()
        args.out_dir = iris_output_dir
        args.output_dir = tmp_path / "figures"
        args.format = "png"
        args.dpi = 100
        args.style = "publication"

        main(args)
        assert (tmp_path / "figures").exists()
        # At least some plots generated
        pngs = list((tmp_path / "figures").glob("*.png"))
        assert len(pngs) >= 3
