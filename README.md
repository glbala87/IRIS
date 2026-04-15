<p align="center">
  <h1 align="center">IRIS</h1>
  <p align="center"><strong>Isoform-Resolved Inference for Single-cells</strong></p>
  <p align="center">
    <a href="#installation">Installation</a> &middot;
    <a href="#quick-start">Quick Start</a> &middot;
    <a href="#modules">Modules</a> &middot;
    <a href="#methodology">Methodology</a> &middot;
    <a href="#validation">Validation</a> &middot;
    <a href="#outputs">Outputs</a>
  </p>
</p>

---

IRIS is a production-ready, isoform-aware single-cell RNA-seq analysis pipeline for long-read data. It reveals cell states invisible to gene-level analysis by clustering cells based on transcript isoform usage, detecting differential transcript usage between clusters, discovering novel isoforms per cell type, and tracking isoform switching along differentiation trajectories.

IRIS accepts standard MEX matrices from **any** upstream preprocessing pipeline (Cell Ranger, STARsolo, wf-single-cell, FLAMES, kallisto|bustools) and produces a unified AnnData `.h5ad` with all results.

**Key features:** reproducible (seeded throughout), parallel (`--n_jobs`), memory-safe (sparse-aware with configurable limits), progress bars, 136 tests passing, validated on PBMC 3k (ARI=0.95, annotation=100%).

## What IRIS does that no other tool does

| Capability | IRIS | Scanpy | Seurat | Cell Ranger | FLAMES | IsoQuant |
|-----------|------|--------|--------|-------------|--------|----------|
| Isoform-usage clustering | **Yes** | No | No | No | No | No |
| Joint gene+isoform embedding (graph fusion) | **Yes** | No | No | No | No | No |
| Differential transcript usage (single-cell) | **Yes** | No | No | No | No | No |
| Isoform switching detection | **Yes** | No | No | No | No | No |
| Cluster-specific novel isoforms | **Yes** | No | No | No | No | No |
| Isoform trajectory switching | **Yes** | No | No | No | No | No |
| Single-cell allele-specific expression | **Yes** | No | No | No | No | No |
| Cell type annotation (20 types, 230+ markers) | **Yes** | Yes | Yes | No | No | No |
| Multi-sample comparison | **Yes** | No | No | No | No | No |
| Unified h5ad export | **Yes** | Yes | No | No | No | No |

## Installation

```bash
# From source
git clone https://github.com/glbala87/IRIS.git
cd IRIS
pip install -e ".[all,dev]"

# With all optional features (ASE, plotting, Leiden)
pip install "iris-sc[all]"

# ASE module only (requires pysam)
pip install "iris-sc[ase]"
```

### Requirements

- Python >= 3.9 (tested on 3.9, 3.10, 3.11, 3.12, 3.13)
- Core: numpy, pandas, scipy, scikit-learn, scanpy, anndata, igraph, umap-learn, tqdm
- Optional: pysam (for ASE), plotly (for reports), leidenalg (for Leiden clustering)

### Conda

```bash
conda env create -f environment.yml
conda activate iris-sc
```

### Docker

```bash
docker build -t iris .
docker run iris --version
```

## Quick Start

### Run the full pipeline (one command)

```bash
iris run \
    gene_matrix/ \
    transcript_matrix/ \
    --gene_transcript_map map.tsv \
    --out_dir iris_results/ \
    --species human \
    --random_state 42 \
    --n_jobs 4
```

This runs all modules in sequence: validation, clustering, dual-layer clustering, cell type annotation, DTU, trajectory, and exports a unified `.h5ad`. Use `--resume` to restart from the last failed step.

### Load results in Python

```python
import scanpy as sc

adata = sc.read_h5ad('iris_results/iris.h5ad')

# Visualize clusters and cell types
sc.pl.umap(adata, color=['joint_cluster', 'cell_type', 'dpt_pseudotime'])

# Access DTU results
dtu = adata.uns['dtu_results']
significant = dtu[dtu['pvalue_adj'] < 0.05]
print(f"{significant['gene'].nunique()} genes with differential transcript usage")

# Access isoform switching events
switching = adata.uns['switching_results']
print(switching[['gene', 'dominant_transcript_a', 'dominant_transcript_b', 'switching_score']].head(10))
```

## Input Requirements

IRIS accepts **standard outputs** from any single-cell preprocessing pipeline. It does **not** do read alignment, barcode extraction, or UMI deduplication.

| Input | Format | Required | Source |
|-------|--------|----------|--------|
| Gene expression matrix | MEX directory (`matrix.mtx.gz`, `barcodes.tsv.gz`, `features.tsv.gz`) | **Yes** | Cell Ranger, STARsolo, wf-single-cell, FLAMES |
| Transcript expression matrix | MEX directory (same format, transcript-level counts) | **Yes** | Any long-read scRNA-seq pipeline |
| Gene-transcript mapping | 2-column TSV (`transcript_id`, `gene_id`) | **Yes** | Derived from GTF or gffcompare |
| Tagged BAM | BAM with CB (cell barcode) tags | Optional | For allele-specific expression |
| Annotated GTFs | gffcompare annotated GTF files | Optional | For novel isoform discovery |
| VCF | Standard VCF with heterozygous variants | Optional | For ASE with known variants |

### Preparing inputs

**Gene-transcript map from GTF:**
```bash
awk -F'\t' '/\ttranscript\t/ {
    match($9, /gene_id "([^"]+)"/, g);
    match($9, /transcript_id "([^"]+)"/, t);
    if (g[1] && t[1]) print t[1]"\t"g[1]
}' genes.gtf | sort -u > gene_transcript_map.tsv
```

## Modules

### Overview

| Command | Description | Key Options |
|---------|-------------|-------------|
| `iris validate` | Check input integrity | |
| `iris dual-cluster` | Gene + isoform dual-layer clustering | `--resolution`, `--isoform_weight`, `--random_state` |
| `iris annotate` | Cell type annotation (20 types, 230+ markers) | `--species`, `--marker_genes_db` |
| `iris dtu` | Differential transcript usage | `--comparison_mode`, `--test_method`, `--n_jobs` |
| `iris novel-isoforms` | Novel isoform discovery | `--enrichment_fdr` |
| `iris trajectory` | Isoform trajectory analysis | `--n_bins`, `--random_state`, `--n_jobs` |
| `iris ase` | Allele-specific expression | `--het_threshold`, `--min_cov`, `--n_jobs` |
| `iris export` | Unified AnnData export | |
| `iris report` | Interactive HTML report | |
| `iris plot` | Publication figures (PNG/PDF/SVG) | `--style`, `--dpi` |
| `iris compare` | Multi-sample comparison | |
| `iris benchmark-dtu` | DTU sensitivity benchmarking | `--effect_sizes`, `--seed` |
| `iris run` | Full pipeline with checkpointing | `--resume`, `--force`, `--n_jobs`, `--random_state` |

---

### `iris dual-cluster` -- Dual-Layer Clustering

Clusters cells at two levels (gene expression and isoform usage), then computes a joint embedding via KNN graph fusion.

```bash
iris dual-cluster gene_matrix/ transcript_matrix/ \
    --gene_transcript_map map.tsv \
    --cluster_method leiden \
    --resolution 0.8 \
    --isoform_weight 3.0 \
    --random_state 42
```

The `--isoform_weight` parameter (default 3.0) controls the balance: at 3.0, the fused graph is 75% isoform / 25% gene.

**Outputs:** `gene_clusters.tsv`, `isoform_clusters.tsv`, `joint_clusters.tsv`, `joint.umap.tsv`, `isoform_diversity.tsv`, `cluster_comparison.json`

---

### `iris dtu` -- Differential Transcript Usage

Tests whether different clusters preferentially use different transcript isoforms. Supports **one-vs-rest** (default) and **all-pairs** comparison modes.

```bash
# One-vs-rest (default)
iris dtu transcript_matrix/ \
    --clusters joint_clusters.tsv \
    --gene_transcript_map map.tsv \
    --n_jobs 8

# All-pairs comparison
iris dtu transcript_matrix/ \
    --clusters joint_clusters.tsv \
    --gene_transcript_map map.tsv \
    --comparison_mode all_pairs \
    --n_jobs 8

# Dirichlet-multinomial test (with convergence diagnostics)
iris dtu transcript_matrix/ \
    --clusters joint_clusters.tsv \
    --gene_transcript_map map.tsv \
    --test_method dirichlet_multinomial \
    --dm_maxiter 500
```

**Methods:** Chi-squared (with configurable Yates' correction) or Dirichlet-multinomial (with convergence diagnostics and automatic fallback).

**Outputs:** `dtu_results.tsv`, `isoform_switching.tsv`, `dtu_summary.json`

---

### `iris annotate` -- Cell Type Annotation

Automated cell type annotation using a curated marker database of **20 cell types** with **230+ markers** (sourced from CellMarker 2.0, PanglaoDB, Human Protein Atlas). Scoring uses specificity-weighted overlap: `fraction_detected x mean_expression x (1 + z_enrichment)`.

```bash
iris annotate gene_matrix/ \
    --clusters joint_clusters.tsv \
    --species human

# Custom markers
iris annotate gene_matrix/ \
    --clusters joint_clusters.tsv \
    --marker_genes_db my_markers.tsv
```

Built-in types include: CD4 T, CD8 T, Treg, naive B, memory B, plasma cells, NK, CD14+ monocytes, FCGR3A+ monocytes, dendritic cells, pDCs, platelets, erythrocytes, macrophages, neutrophils, mast cells, fibroblasts, epithelial, endothelial, smooth muscle.

**Outputs:** `cell_type_annotations.tsv`, `cluster_cell_types.tsv`, `cell_type_summary.json`

---

### `iris trajectory` -- Isoform Trajectory Analysis

Computes diffusion pseudotime, then tracks isoform proportion dynamics along the trajectory.

```bash
iris trajectory gene_matrix/ \
    --transcript_matrix_dir transcript_matrix/ \
    --gene_transcript_map map.tsv \
    --random_state 42 \
    --n_jobs 4
```

**Outputs:** `pseudotime.tsv`, `isoform_dynamics.tsv`, `trajectory_switching.tsv`, `trajectory_summary.json`

---

### `iris ase` -- Allele-Specific Expression

Detects allelic imbalance at heterozygous sites using long-read phasing.

> **Note:** Requires `pip install "iris-sc[ase]"` for pysam.

```bash
iris ase tagged.bam \
    --clusters joint_clusters.tsv \
    --vcf variants.vcf.gz \
    --het_threshold 0.20 \
    --min_cov 10 \
    --n_jobs 4
```

**Outputs:** `ase_results.tsv`, `ase_results_differential.tsv`, `ase_summary.json`

---

### `iris plot` -- Publication Figures

```bash
iris plot --out_dir iris_results/ --format pdf --dpi 300 --style publication
```

Generates: `umap_clusters`, `umap_cell_types`, `umap_diversity`, `dtu_volcano`, `isoform_heatmap`, `trajectory_pseudotime`, `novel_class_codes`, `ase_manhattan`, `cluster_contingency`. Styles: `publication` (Nature/Cell) or `presentation`.

## Methodology

### Dual-Layer Clustering with Graph Fusion

```
Gene Expression Matrix              Transcript Expression Matrix
        |                                    |
        v                                    v
   Normalize, HVG, PCA              Isoform usage proportions, PCA
        |                                    |
        v                                    v
   Gene KNN graph                    Isoform KNN graph
        |                                    |
        +--------- Graph Fusion ------------+
        |     W = (1/(1+w)) * W_gene        |
        |       + (w/(1+w)) * W_iso         |
        v                                    v
   Gene clusters                    Isoform clusters
        |                                    |
        +------------ Compare (ARI, NMI) ---+
                         |
                         v
                  Joint Embedding (UMAP on fused graph)
```

### Memory Safety

All sparse-to-dense conversions are routed through `safe_toarray()` with a configurable memory guard (default 2 GB, set via `--max_dense_gb`).

## Pipeline Features

| Feature | Description |
|---------|-------------|
| **Checkpointing** | `--resume` restarts from last failed step, `--force` re-runs everything |
| **Error recovery** | Failed modules don't crash the pipeline; partial results are saved |
| **Reproducibility** | `--random_state` seeds all stochastic operations (PCA, clustering, UMAP) |
| **Parallelization** | `--n_jobs` for DTU, ASE, and trajectory modules |
| **Progress bars** | tqdm progress bars for long-running operations (auto-detect terminal) |
| **Configurable limits** | `--max_dense_gb`, `--het_threshold`, `--dm_maxiter`, `--yates_correction` |
| **Convergence diagnostics** | DM test logs fallback warnings; fallback count reported in summary |

## Validation

IRIS includes a **reproducible validation script** that downloads PBMC 3k data, generates synthetic transcripts, and benchmarks all modules:

```bash
python validation/run_pbmc3k_validation.py --output_dir validation/pbmc3k_results/
```

### PBMC 3k Benchmark Results (2,638 cells, 1,838 genes, 5,518 transcripts)

| Metric | Result |
|--------|--------|
| Pipeline steps completed | **6/6** (ASE and novel isoforms require BAM/GTF) |
| Gene clusters | 5 |
| Isoform clusters | 8-9 |
| Joint clusters | 6-8 |
| Joint clustering ARI vs known cell types | **0.95** |
| Joint clustering NMI vs known cell types | **0.94** |
| Isoform clustering ARI vs known cell types | **0.998** |
| Cell type annotation | **8/8 clusters annotated correctly** |
| Cell types identified | CD4 T, CD8 T, Naive B, NK, CD14+ Mono, FCGR3A+ Mono, DC, Platelets |
| DTU genes (FDR < 0.05) | 1,829 |
| Isoform switching events | 7,988 |
| Dynamic isoforms along trajectory | 3,267 |
| Trajectory switching genes | 14 |
| Pipeline runtime | ~30 seconds |

### DTU Benchmark (sensitivity/specificity)

```bash
iris benchmark-dtu --output_dir benchmark/ --seed 42
```

| Effect size | Sensitivity | Specificity | F1 |
|-------------|-------------|-------------|-----|
| 0.1 | 0.567 | 0.992 | 0.688 |
| 0.3 | 1.000 | 0.983 | 0.970 |
| 0.5 | 1.000 | 1.000 | 1.000 |
| 0.7 | 1.000 | 0.983 | 0.970 |
| 0.9 | 1.000 | 0.992 | 0.984 |

### DRIMSeq Comparison

```bash
python validation/benchmark_vs_drimseq.py --output_dir validation/benchmark_results/
```

Compares IRIS DTU detection against DRIMSeq (R package) on synthetic data with known ground truth. Requires R with DRIMSeq installed; runs IRIS-only if R is unavailable.

### Known Limitations

- **Sparse scaling warning.** Scanpy's `sc.pp.scale()` densifies sparse matrices internally. For datasets >50k cells, subsample or reduce features first.
- **ASE requires pysam.** Install with `pip install "iris-sc[ase]"`. Does not require MD tags.
- **Custom markers recommended** for non-PBMC tissues. The built-in database covers common blood and tissue types.

## Nextflow

IRIS includes a Nextflow wrapper for HPC/cloud execution:

```bash
nextflow run IRIS/nextflow/main.nf \
    --gene_matrix_dir gene_matrix/ \
    --transcript_matrix_dir transcript_matrix/ \
    --gene_transcript_map map.tsv \
    --out_dir iris_results/ \
    --species human \
    -profile docker
```

Profiles: `conda`, `docker`, `singularity`. Each process uses dynamic memory allocation with automatic OOM retry.

## CLI Reference

```
$ iris --help

IRIS: Isoform-Resolved Inference for Single-cells.

commands:
  cluster          Basic Scanpy clustering
  dual-cluster     Gene + isoform dual-layer clustering
  dtu              Differential transcript usage testing
  novel-isoforms   Cluster-specific novel isoform discovery
  trajectory       Isoform-aware trajectory analysis
  ase              Allele-specific expression analysis
  annotate         Cell type annotation
  export           Export unified AnnData h5ad
  benchmark-dtu    DTU detection benchmarking
  report           Generate HTML report
  plot             Generate publication-quality figures
  compare          Cross-sample comparison
  validate         Validate pipeline inputs
  run              Run full pipeline
```

## Outputs

### Per-sample outputs (from `iris run`)

| File | Description |
|------|-------------|
| `iris.h5ad` | Unified AnnData with all results |
| `gene_clusters.tsv` | Gene-level cluster assignments |
| `isoform_clusters.tsv` | Isoform-usage cluster assignments |
| `joint_clusters.tsv` | Joint cluster assignments |
| `joint.umap.tsv` | Joint UMAP coordinates |
| `isoform_diversity.tsv` | Per-cell Shannon/Simpson diversity |
| `cluster_comparison.json` | ARI, NMI, isoform-specific clusters |
| `cell_type_annotations.tsv` | Per-cell type labels |
| `cluster_cell_types.tsv` | Per-cluster type summary |
| `dtu_results.tsv` | DTU test results per gene per cluster |
| `isoform_switching.tsv` | Detected isoform switching events |
| `pseudotime.tsv` | Diffusion pseudotime per cell |
| `isoform_dynamics.tsv` | Isoform trend statistics along trajectory |
| `trajectory_switching.tsv` | Trajectory switching events |
| `iris.log` | Full pipeline log |
| `iris_pipeline_summary.json` | Pipeline execution summary |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=iris --cov-report=term-missing

# Current: 136 passed, 2 skipped
```

## License

MIT

## Citation

If you use IRIS in your research, please cite:

> IRIS: Isoform-Resolved Inference for Single-cells — isoform-aware analysis for long-read sequencing data. (2026).
