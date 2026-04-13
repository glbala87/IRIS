<p align="center">
  <h1 align="center">IRIS</h1>
  <p align="center"><strong>Isoform-Resolved Inference for Single-cells</strong></p>
  <p align="center">
    <a href="#installation">Installation</a> &middot;
    <a href="#quick-start">Quick Start</a> &middot;
    <a href="#modules">Modules</a> &middot;
    <a href="#methodology">Methodology</a> &middot;
    <a href="#outputs">Outputs</a>
  </p>
</p>

---

IRIS is a standalone, isoform-aware single-cell RNA-seq analysis tool for long-read data. It reveals cell states invisible to gene-level analysis by clustering cells based on transcript isoform usage, detecting differential transcript usage between clusters, discovering novel isoforms per cell type, and tracking isoform switching along differentiation trajectories.

IRIS accepts standard MEX matrices from **any** upstream preprocessing pipeline (Cell Ranger, STARsolo, wf-single-cell, FLAMES, kallisto|bustools) and produces a unified AnnData `.h5ad` with all results.

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
| Cell type annotation | **Yes** | Yes | Yes | No | No | No |
| Multi-sample comparison | **Yes** | No | No | No | No | No |
| Unified h5ad export | **Yes** | Yes | No | No | No | No |

## Installation

```bash
# From PyPI
pip install iris-sc

# With all optional features (ASE, plotting, Leiden)
pip install "iris-sc[all]"

# ASE module only (requires pysam)
pip install "iris-sc[ase]"

# Development install from source
git clone https://github.com/iris-sc/iris.git
cd iris
pip install -e ".[all,dev]"
```

### Requirements

- Python >= 3.9 (tested on 3.9, 3.10, 3.11, 3.12, 3.13)
- Core: numpy, pandas, scipy, scikit-learn, scanpy, anndata, igraph, umap-learn
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
    --species human
```

This runs all modules in sequence: validation, clustering, dual-layer clustering, cell type annotation, DTU, trajectory, and exports a unified `.h5ad`.

### Load results in Python

```python
import scanpy as sc

adata = sc.read_h5ad('iris_results/iris.h5ad')

# Visualize clusters and cell types
sc.pl.umap(adata, color=['gene_cluster', 'isoform_cluster', 'cell_type'])

# Access DTU results
dtu = adata.uns['dtu_results']
significant = dtu[dtu['pvalue_adj'] < 0.05]
print(f"{significant['gene'].nunique()} genes with differential transcript usage")

# Access isoform switching events
switching = adata.uns['switching_results']
print(switching[['gene', 'dominant_transcript_a', 'dominant_transcript_b', 'switching_score']].head(10))

# Per-cell metadata
print(adata.obs[['gene_cluster', 'isoform_cluster', 'cell_type',
                  'diversity_index', 'dpt_pseudotime']].describe())
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

**From Cell Ranger output:**
```bash
iris run \
    cellranger_output/outs/filtered_feature_bc_matrix/ \
    transcript_matrix/ \
    --gene_transcript_map map.tsv \
    --out_dir results/
```

**From wf-single-cell output:**
```bash
iris run \
    wf_output/gene_processed_feature_bc_matrix/ \
    wf_output/transcript_processed_feature_bc_matrix/ \
    --gene_transcript_map map.tsv \
    --tagged_bam wf_output/tagged.bam \
    --annotated_gtfs wf_output/gffcompare.*.gtf \
    --out_dir results/
```

## Modules

### Overview

| Command | Description | Inputs | Key Outputs |
|---------|-------------|--------|-------------|
| `iris validate` | Check input integrity | MEX dirs, TSVs, BAM | Pass/fail report |
| `iris dual-cluster` | Gene + isoform dual-layer clustering | Gene MEX, Transcript MEX, map | Clusters, UMAP, diversity |
| `iris annotate` | Cell type annotation | Gene MEX, clusters | Cell type labels |
| `iris dtu` | Differential transcript usage | Transcript MEX, clusters, map | DTU results, switching events |
| `iris novel-isoforms` | Novel isoform discovery | Transcript MEX, clusters, GTFs | Novel catalog, enrichment |
| `iris trajectory` | Isoform trajectory analysis | Gene MEX, Transcript MEX, map | Pseudotime, dynamics |
| `iris ase` | Allele-specific expression | Tagged BAM, clusters | ASE results |
| `iris export` | Unified AnnData export | All outputs | `.h5ad` file |
| `iris report` | HTML report | Output directory | Interactive HTML |
| `iris plot` | Publication figures | Output directory | PNG/PDF/SVG |
| `iris compare` | Multi-sample comparison | Multiple output dirs | Cross-sample results |
| `iris benchmark-dtu` | DTU sensitivity benchmarking | Parameters | Benchmark results |
| `iris run` | Full pipeline | All inputs | All outputs |

---

### `iris dual-cluster` -- Dual-Layer Clustering

Clusters cells at two independent levels, then computes a joint embedding via KNN graph fusion:

1. **Gene expression clustering** -- standard Scanpy pipeline (normalize, HVG, PCA, KNN, Louvain/Leiden, UMAP)
2. **Isoform usage clustering** -- computes per-cell transcript proportions within each gene, then clusters by these splicing patterns (scale, PCA, KNN, Louvain/Leiden, UMAP)
3. **Joint embedding** -- builds separate KNN graphs in gene and isoform PCA space, then merges them as a weighted combination of connectivity matrices (WNN-inspired graph fusion). The `--isoform_weight` parameter controls the relative influence of isoform-level structure.
4. **Cluster comparison** -- computes ARI and NMI between gene-level and isoform-level clusterings, identifies isoform-specific clusters

```bash
iris dual-cluster gene_matrix/ transcript_matrix/ \
    --gene_transcript_map map.tsv \
    --cluster_method louvain \
    --resolution 0.8 \
    --isoform_resolution 1.0 \
    --isoform_weight 3.0 \
    --n_neighbors 15 \
    --n_pcs 50 \
    --diversity_metric shannon
```

The `--isoform_weight` parameter (default 3.0) controls the balance between gene expression and isoform usage in the joint embedding. At 3.0, the fused graph is 75% isoform / 25% gene. Increase this when isoform usage is the primary driver of cell-state differences; decrease when gene expression is more informative.

**Outputs:** `gene_clusters.tsv`, `isoform_clusters.tsv`, `joint_clusters.tsv`, `joint.umap.tsv`, `isoform_diversity.tsv`, `cluster_comparison.json`

---

### `iris dtu` -- Differential Transcript Usage

Tests whether different cell clusters preferentially use different transcript isoforms for the same gene, even when total gene expression is unchanged.

**Methods:**
- **Chi-squared test** (default) -- builds a 2 x n_isoforms contingency table per gene, tests independence of cluster identity and isoform choice. Uses Yates' continuity correction for robustness with small counts.
- **Dirichlet-multinomial test** -- fits DM models per cluster, likelihood ratio test for differential isoform proportions. Handles overdispersion in single-cell data.

**Isoform switching:** for each significant DTU gene, checks if the dominant transcript changes between clusters.

```bash
iris dtu transcript_matrix/ \
    --clusters joint_clusters.tsv \
    --gene_transcript_map map.tsv \
    --test_method chi_squared \
    --fdr_threshold 0.05 \
    --min_cells_per_cluster 10 \
    --min_gene_counts 20
```

**Outputs:** `dtu_results.tsv`, `isoform_switching.tsv`, `dtu_summary.json`

---

### `iris novel-isoforms` -- Novel Isoform Discovery

Identifies novel transcript isoforms from StringTie/gffcompare assemblies that are enriched in specific cell clusters.

Uses gffcompare class codes:
- `j` -- novel junction-sharing isoform (shares splice site with reference)
- `u` -- intergenic (completely novel, no reference overlap)
- `o` -- generic exonic overlap with reference
- `x` -- exonic overlap on opposite strand (antisense)

For each novel isoform, tests cluster enrichment using Fisher's exact test, then computes a specificity score: `log2(fold_enrichment + 1) * (1 - normalized_entropy)`. This preserves fold-change magnitude for moderately distributed isoforms while penalizing broadly expressed ones.

```bash
iris novel-isoforms transcript_matrix/ \
    --clusters joint_clusters.tsv \
    --gene_transcript_map map.tsv \
    --annotated_gtfs gffcompare.chr*.gtf \
    --min_cells 3 \
    --min_counts 5 \
    --enrichment_fdr 0.05
```

**Outputs:** `novel_isoform_catalog.tsv`, `novel_isoform_enrichment.tsv`, `novel_isoform_summary.json`

---

### `iris annotate` -- Cell Type Annotation

Automated cell type annotation using marker gene databases.

**Methods:**
- **Marker overlap** (default) -- for each cluster, counts detected marker genes per cell type, scores by n_detected x mean_expression
- **Correlation** -- correlates cluster mean expression profiles with binary reference profiles (Pearson)

Built-in marker databases for human and mouse (12 cell types: T cells, B cells, NK cells, monocytes, dendritic cells, platelets, erythrocytes, fibroblasts, epithelial, endothelial, macrophages, neutrophils). Custom marker TSV supported.

```bash
iris annotate gene_matrix/ \
    --clusters joint_clusters.tsv \
    --species human \
    --method marker_overlap \
    --min_marker_genes 3

# Custom markers
iris annotate gene_matrix/ \
    --clusters joint_clusters.tsv \
    --marker_genes_db my_markers.tsv
```

**Outputs:** `cell_type_annotations.tsv`, `cluster_cell_types.tsv`, `cell_type_summary.json`

---

### `iris trajectory` -- Isoform Trajectory Analysis

Computes diffusion pseudotime from gene expression, then tracks how isoform usage proportions change along the trajectory.

1. **Diffusion pseudotime** -- normalize, HVG, PCA, diffusion map, DPT (auto-selects root cell)
2. **Isoform trends** -- bins cells by pseudotime, computes aggregate isoform proportions per bin, Spearman correlation with pseudotime
3. **Trajectory switching** -- identifies genes where opposing isoform trends indicate a dominant transcript change during differentiation

```bash
iris trajectory gene_matrix/ \
    --transcript_matrix_dir transcript_matrix/ \
    --gene_transcript_map map.tsv \
    --n_dpt_neighbors 15 \
    --n_pcs 30 \
    --n_bins 10
```

**Outputs:** `pseudotime.tsv`, `isoform_dynamics.tsv`, `trajectory_switching.tsv`, `trajectory_summary.json`

---

### `iris ase` -- Allele-Specific Expression

Detects allelic imbalance at heterozygous sites using long-read phasing. Long reads span multiple SNPs per cell, enabling haplotype-resolved expression analysis.

> **Note:** Requires `pip install "iris-sc[ase]"` for pysam. Does not require MD tags in the BAM file.

1. **Variant discovery** -- scans BAM pileups for het positions (or loads from VCF)
2. **Per-cell allele counting** -- counts ref/alt alleles per cell per variant using CB tags
3. **Imbalance testing** -- binomial test (H0: balanced, p=0.5) per variant per cluster
4. **Differential ASE** -- Fisher's exact test for allelic ratio differences between clusters

```bash
iris ase tagged.bam \
    --clusters joint_clusters.tsv \
    --vcf variants.vcf.gz \
    --min_total_counts 10 \
    --min_cells 5 \
    --fdr_threshold 0.05
```

**Outputs:** `ase_results.tsv`, `ase_results_differential.tsv`, `ase_summary.json`

---

### `iris compare` -- Multi-Sample Comparison

Compares results across multiple samples to identify conserved vs sample-specific findings.

- **Conserved DTU** -- genes with significant DTU in all samples
- **Shared novel isoforms** -- novel transcripts present across samples
- **Conserved switching** -- same isoform switch in multiple samples
- **Cell type composition** -- Fisher's exact test for differential composition

```bash
iris compare \
    --sample_dirs patient1/iris_results/ patient2/iris_results/ \
    --sample_names patient1 patient2 \
    --output_dir comparison/
```

**Outputs:** `conserved_dtu.tsv`, `sample_specific_dtu.tsv`, `shared_novel_isoforms.tsv`, `composition_comparison.tsv`, `conserved_switches.tsv`, `comparison_summary.json`

---

### `iris plot` -- Publication Figures

Generates publication-quality figures from IRIS results.

```bash
iris plot \
    --out_dir iris_results/ \
    --output_dir iris_figures/ \
    --format pdf \
    --dpi 300 \
    --style publication
```

**Figures generated:**
| File | Description |
|------|-------------|
| `umap_clusters.pdf` | Joint UMAP colored by cluster |
| `umap_cell_types.pdf` | Joint UMAP colored by cell type |
| `umap_diversity.pdf` | Joint UMAP colored by isoform diversity |
| `dtu_volcano.pdf` | Volcano plot of DTU effect size vs significance |
| `isoform_heatmap.pdf` | Heatmap of isoform proportions for top switching genes |
| `trajectory_pseudotime.pdf` | UMAP colored by pseudotime |
| `novel_class_codes.pdf` | Bar plot of novel isoform class codes |
| `ase_manhattan.pdf` | Manhattan plot of ASE significance |
| `cluster_contingency.pdf` | Gene vs isoform cluster contingency heatmap |

Styles: `publication` (Nature/Cell formatting) or `presentation` (larger fonts).

---

### `iris export` -- AnnData Export

Exports all results into a single `.h5ad` file for downstream analysis in Scanpy or any AnnData-compatible tool.

```bash
iris export gene_matrix/ \
    --transcript_matrix_dir transcript_matrix/ \
    --joint_clusters joint_clusters.tsv \
    --joint_umap joint.umap.tsv \
    --cell_type_annotations cell_type_annotations.tsv \
    --isoform_diversity isoform_diversity.tsv \
    --dtu_results dtu_results.tsv \
    --switching_results isoform_switching.tsv \
    --pseudotime pseudotime.tsv \
    --output iris.h5ad
```

**h5ad structure:**
```
adata.X                  -> gene expression matrix (sparse)
adata.layers['counts']   -> raw counts
adata.obs                -> gene_cluster, isoform_cluster, joint_cluster,
                           cell_type, diversity_index, dpt_pseudotime
adata.obsm['X_umap']     -> joint UMAP embedding
adata.uns['dtu_results'] -> DTU test results DataFrame
adata.uns['switching_results'] -> isoform switching events
adata.uns['cluster_comparison'] -> ARI/NMI comparison stats (JSON string)
```

## Methodology

### Dual-Layer Clustering with Graph Fusion

```
Gene Expression Matrix              Transcript Expression Matrix
(cells x genes)                     (cells x transcripts)
        |                                    |
        v                                    v
   Normalize                         Compute isoform usage
   Log1p                             proportions per gene
   HVG selection                     (transcript / gene total)
        |                                    |
        v                                    v
      PCA                                  PCA
   (gene space)                      (isoform usage space)
        |                                    |
        v                                    v
   Gene KNN graph                    Isoform KNN graph
        |                                    |
        +--------- Graph Fusion ------------+
        |     (weighted combination of       |
        |      connectivity matrices)        |
        v                                    v
   Gene clusters                    Isoform clusters
   (Louvain/Leiden)                 (Louvain/Leiden)
        |                                    |
        +------------ Compare --------------+
                   ARI, NMI
                   Isoform-specific clusters
                         |
                         v
                  Joint Embedding
             (cluster + UMAP on fused graph)
```

The joint embedding uses **KNN graph fusion** inspired by Seurat v4 WNN (Hao et al., Cell 2021). Instead of concatenating PCA spaces (which suffers from the curse of dimensionality when one modality carries noise), separate KNN graphs are built in gene and isoform space. The graphs are merged as a weighted combination:

```
W_joint = (1 / (1 + w)) * W_gene + (w / (1 + w)) * W_iso
```

where `w` is the `--isoform_weight` parameter (default 3.0, giving 75% isoform / 25% gene). This preserves cell-state structure from whichever modality defines it.

**Key insight:** Cells that express the same genes at the same level can still differ in *how* they splice those genes. Isoform-usage clustering captures this splicing-driven heterogeneity that gene-level analysis misses.

### Isoform Diversity Index

For each cell, for each multi-isoform gene:
- Shannon entropy: H = -sum(p * log2(p)) where p = proportion of each isoform
- Averaged across all expressed multi-isoform genes

High diversity = cell uses many isoforms roughly equally.
Low diversity = cell preferentially uses one dominant isoform.

### Differential Transcript Usage

For each gene with >= 2 expressed isoforms, tests whether isoform proportions differ between clusters:

**Chi-squared test (with Yates' continuity correction):**
```
                  Isoform A    Isoform B    Isoform C
Cluster 1:          120           30            50
Cluster 2:           40          100            60
                            |
              chi2 contingency test -> p-value -> BH correction
```

Effect size: Cramer's V. Multiple testing correction: Benjamini-Hochberg FDR.

**Isoform switching:** detected when the dominant transcript changes between clusters (e.g., Gene X uses isoform A in T cells but isoform B in B cells).

### Novel Isoform Discovery

```
StringTie assembly -> gffcompare vs reference -> class codes
                                                    |
              +-------------------------------------+
              v                                     v
         Known (=, c)                        Novel (j, u, o, x)
                                                    |
                                                    v
                                      Cross-reference with
                                      transcript expression matrix
                                                    |
                                                    v
                                      Fisher's exact test for
                                      cluster enrichment
                                                    |
                                                    v
                                      Specificity score:
                                      log2(fold+1) * (1 - entropy)
```

### Memory Safety

All sparse-to-dense matrix conversions include a memory guard (default 2 GB limit). If densifying a matrix would exceed this limit, a clear `MemoryError` is raised with the estimated size.

## Pipeline Features

### Checkpointing

```bash
# Pipeline saves progress after each step
iris run ... --out_dir results/

# If step 4 fails, fix the issue and resume from step 4
iris run ... --out_dir results/ --resume

# Force re-run everything
iris run ... --out_dir results/ --force
```

### Error Recovery

Each module is wrapped in error handling. If one module fails, the pipeline:
- Logs the error with full traceback
- Marks the step as failed in `.iris_checkpoint.json`
- Continues to the next module
- Saves partial results
- Reports a summary at the end

### Input Validation

```bash
# Pre-flight check before running
iris validate \
    --gene_matrix_dir gene_matrix/ \
    --transcript_matrix_dir transcript_matrix/ \
    --gene_transcript_map map.tsv \
    --tagged_bam tagged.bam
```

Checks:
- MEX directory structure (matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz)
- Barcode overlap between gene and transcript matrices
- Duplicate barcode detection
- BAM index existence and CB tag presence
- TSV column validation

### Logging

All output goes to both stderr and `iris.log` in the output directory:

```
[02:00:01 - iris.DualClust] Gene matrix: 2700 cells x 32738 genes.
[02:00:04 - iris.Compare  ] ARI=0.2343, NMI=0.4789 over 2700 cells.
[02:00:04 - iris.Compare  ] Found 1 isoform-specific clusters.
```

## Validation

IRIS has been validated on the PBMC 3k benchmark dataset (2,700 cells, 32,738 genes, 12,480 transcript isoforms):

| Metric | Result |
|--------|--------|
| Pipeline steps completed | **8/8** |
| Gene clusters | 8 |
| Isoform clusters | 15 |
| Joint clusters | 5-7 |
| ARI (gene vs isoform) | 0.23 (confirms distinct structure) |
| Isoform clusters vs ground truth | ARI = 0.49, NMI = 0.71 |
| Isoform-specific clusters | 1 |
| Cell types annotated | T cells (1,211), NK cells (432), B cells (344) |
| DTU genes (FDR < 0.05) | 280 |
| Isoform switching events | 16,468 |
| Novel isoforms cataloged | 4,992 |
| Cluster-enriched novel isoforms | 2,479 |
| Dynamic isoforms along trajectory | 3,996 |
| ASE variants tested | 200 input |
| Pipeline runtime | ~3 minutes (2,700 cells) |

### Known Limitations

- **Validated on 2,700-cell dataset.** Large datasets (100k+ cells) will hit the 2 GB memory guard on matrix densification -- reduce features or cells first.
- **Joint embedding vs isoform-only.** When gene expression carries no cell-state signal, isoform-only clusters will outperform the joint embedding. This is expected -- the joint embedding is most valuable when both modalities contribute information.
- **ASE requires pysam.** Install with `pip install "iris-sc[ase]"`. The module does not require MD tags in the BAM.
- **Chi-squared DTU uses Yates' continuity correction.** This is more conservative than an uncorrected test, which is appropriate for the sparse contingency tables common in single-cell data.

## Nextflow

IRIS includes a Nextflow wrapper for HPC/cloud execution with automatic memory scaling and OOM retry:

```bash
nextflow run iris/nextflow/main.nf \
    --gene_matrix_dir gene_matrix/ \
    --transcript_matrix_dir transcript_matrix/ \
    --gene_transcript_map map.tsv \
    --out_dir iris_results/ \
    --species human \
    --differential_transcript_usage true \
    --novel_isoform_discovery true \
    --isoform_trajectory true \
    -profile conda
```

Profiles: `conda`, `docker`, `singularity`

Each process uses dynamic memory allocation (`memory { base * task.attempt }`) with automatic retry on OOM (exit codes 137, 140).

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

Each command supports `--help`:
```bash
iris dtu --help
iris dual-cluster --help
iris run --help
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
| `novel_isoform_catalog.tsv` | Novel isoform catalog |
| `novel_isoform_enrichment.tsv` | Cluster enrichment of novel isoforms |
| `pseudotime.tsv` | Diffusion pseudotime per cell |
| `isoform_dynamics.tsv` | Isoform trend statistics along trajectory |
| `trajectory_switching.tsv` | Trajectory switching events |
| `ase_results.tsv` | Allele-specific expression results |
| `clusters.tsv` | Basic Scanpy cluster assignments |
| `marker_genes.tsv` | Marker genes per cluster |
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
