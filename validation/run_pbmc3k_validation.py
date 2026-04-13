#!/usr/bin/env python3
"""
Reproducible validation of the IRIS pipeline against the PBMC 3k benchmark dataset.

Downloads the Scanpy-processed PBMC 3k dataset (Zheng et al. 2017), generates
synthetic transcript-level count data with Dirichlet-distributed isoform
proportions, then runs the IRIS dual-layer clustering, cell type annotation,
DTU, trajectory and export modules. Clustering quality is evaluated against
the known Louvain labels using ARI and NMI; cell type annotation accuracy is
reported as fraction of clusters correctly labelled.

Usage
-----
    python validation/run_pbmc3k_validation.py \\
        --output_dir validation/pbmc3k_results/ \\
        --seed 42

Output
------
    <output_dir>/
        gene_matrix/          – MEX matrices for gene-level counts
        transcript_matrix/    – MEX matrices for synthetic transcript counts
        gene_transcript_map.tsv
        iris_output/          – all IRIS pipeline outputs
        validation_report.json
"""

import argparse
import gzip
import json
import logging
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PBMC3kValidation")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Reproducible IRIS validation on PBMC 3k dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("validation/pbmc3k_results/"),
        help="Directory to write all outputs and the validation report.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible synthetic transcript generation.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step 1: Download PBMC 3k processed data
# ---------------------------------------------------------------------------

def download_pbmc3k():
    """Download the Scanpy-processed PBMC 3k dataset.

    Returns the AnnData object.  The dataset contains 2638 PBMCs with
    Louvain cluster labels in adata.obs['louvain'].
    """
    import scanpy as sc

    logger.info("Downloading PBMC 3k processed dataset via scanpy...")
    adata = sc.datasets.pbmc3k_processed()
    logger.info(
        f"  Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
    logger.info(
        f"  Louvain clusters: {sorted(adata.obs['louvain'].unique())}")
    return adata


# ---------------------------------------------------------------------------
# Step 2: Save gene matrix in MEX format
# ---------------------------------------------------------------------------

def _write_mex(out_dir: Path, matrix, barcodes, feature_names):
    """Write a sparse matrix in Cell Ranger MEX format (gzipped).

    Parameters
    ----------
    out_dir:
        Directory to create (will be created if absent).
    matrix:
        scipy sparse matrix, shape (n_features, n_cells) — i.e., genes/
        transcripts as rows, cells as columns (Cell Ranger convention).
    barcodes:
        Sequence of barcode strings, length == n_cells.
    feature_names:
        Sequence of feature name strings, length == n_features.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # matrix.mtx.gz – Cell Ranger stores features x cells
    mtx_path = out_dir / "matrix.mtx"
    scipy.io.mmwrite(str(mtx_path), matrix)
    with open(mtx_path, "rb") as f_in, \
            gzip.open(out_dir / "matrix.mtx.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    mtx_path.unlink()

    # barcodes.tsv.gz
    bc_path = out_dir / "barcodes.tsv"
    with open(bc_path, "w") as fh:
        for bc in barcodes:
            fh.write(f"{bc}\n")
    with open(bc_path, "rb") as f_in, \
            gzip.open(out_dir / "barcodes.tsv.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    bc_path.unlink()

    # features.tsv.gz  (name, name, "Gene Expression")
    feat_path = out_dir / "features.tsv"
    with open(feat_path, "w") as fh:
        for name in feature_names:
            fh.write(f"{name}\t{name}\tGene Expression\n")
    with open(feat_path, "rb") as f_in, \
            gzip.open(out_dir / "features.tsv.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    feat_path.unlink()

    logger.info(
        f"  MEX written to {out_dir}  "
        f"({len(barcodes)} cells x {len(feature_names)} features)")


def save_gene_matrix(adata, out_dir: Path):
    """Write the PBMC 3k gene expression counts as a MEX matrix.

    Scanpy's pbmc3k_processed() stores log-normalised values in adata.X;
    we recover approximate raw counts by reversing log1p and renormalising
    to integers so that downstream IRIS modules receive integer counts.

    Parameters
    ----------
    adata : AnnData
        Processed PBMC 3k object.
    out_dir : Path
        Destination directory for gene MEX files.

    Returns
    -------
    list[str]
        Barcode list (in order).
    list[str]
        Gene name list (in order).
    """
    import scipy.sparse as sp

    logger.info("Writing gene MEX matrix...")

    # Reverse log1p to get approximate normalised counts, then scale to
    # integers.  adata.X may be dense or sparse.
    X = adata.X
    if sp.issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X)

    # expm1 reverses log1p; values are in the range [0, ~10] after
    # normalisation to 10 000 counts/cell and log1p.
    X_counts = np.expm1(X_dense)

    # Round to nearest integer and clip negatives (floating-point artefacts)
    X_counts = np.round(X_counts).clip(0).astype(np.float32)

    barcodes = list(adata.obs_names)
    gene_names = list(adata.var_names)

    # MEX convention: features x cells (transpose of AnnData's cells x genes)
    matrix = scipy.sparse.csc_matrix(X_counts.T)
    _write_mex(out_dir, matrix, barcodes, gene_names)

    return barcodes, gene_names


# ---------------------------------------------------------------------------
# Step 3: Generate synthetic transcript-level data
# ---------------------------------------------------------------------------

def generate_transcript_data(
        adata, barcodes, gene_names, out_dir: Path,
        map_out: Path, seed: int = 42):
    """Synthesise per-transcript count matrices from gene-level counts.

    For every gene we draw 2–4 synthetic transcript IDs and distribute
    the gene's counts across those transcripts according to Dirichlet-
    sampled proportions.  Proportions are cell-type-aware: each Louvain
    cluster gets its own Dirichlet draw so that DTU signal is present.

    Parameters
    ----------
    adata : AnnData
        Processed PBMC 3k object (used for Louvain labels).
    barcodes : list[str]
        Cell barcodes in matrix row order.
    gene_names : list[str]
        Gene names in matrix column order.
    out_dir : Path
        Destination directory for transcript MEX files.
    map_out : Path
        Where to write the gene_transcript_map.tsv.
    seed : int
        Random seed.

    Returns
    -------
    list[str]
        Transcript IDs in output matrix column order.
    dict[str, str]
        Mapping transcript_id -> gene_id.
    """
    import scipy.sparse as sp

    logger.info(
        f"Generating synthetic transcript data (seed={seed})...")
    rng = np.random.default_rng(seed)

    # ----- Reconstruct integer gene counts --------------------------------
    X = adata.X
    if sp.issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X, dtype=float)
    X_counts = np.round(np.expm1(X_dense)).clip(0).astype(np.int32)
    # shape: (n_cells, n_genes)

    n_cells = X_counts.shape[0]
    n_genes = len(gene_names)

    # ----- Assign transcript count per gene (2–4 isoforms) ----------------
    n_isoforms_per_gene = rng.integers(2, 5, size=n_genes)  # [2, 4]

    # Build transcript ID list and gene->transcript index map
    transcript_ids = []
    gene_to_tx_indices = {}   # gene_name -> [col indices in tx matrix]
    tx_to_gene = {}           # transcript_id -> gene_id

    for gi, gene in enumerate(gene_names):
        n_iso = int(n_isoforms_per_gene[gi])
        indices = []
        for t in range(n_iso):
            tx_id = f"{gene}.iso{t + 1}"
            transcript_ids.append(tx_id)
            tx_to_gene[tx_id] = gene
            indices.append(len(transcript_ids) - 1)
        gene_to_tx_indices[gene] = indices

    n_transcripts = len(transcript_ids)
    logger.info(
        f"  {n_genes} genes -> {n_transcripts} synthetic transcripts "
        f"(mean {n_transcripts / n_genes:.1f} per gene)")

    # ----- Louvain cluster memberships ------------------------------------
    # Map each cell to its Louvain cluster so we can draw cluster-specific
    # Dirichlet proportions.
    louvain = adata.obs["louvain"].values
    clusters = sorted(set(louvain))

    # Pre-draw Dirichlet concentration parameters (alpha) per gene per
    # cluster.  We use asymmetric alpha to create realistic isoform
    # switching patterns between clusters.
    #   - alpha ~ Dirichlet(1, 1, ...) gives uniform proportions on average
    #   - We perturb with a random gamma draw to break symmetry
    alpha_base = np.ones(4)  # max 4 isoforms
    cluster_props = {}  # cluster -> (n_genes, max_iso) float array
    for cl in clusters:
        # Draw a random dominant isoform per gene for this cluster
        # shape: (n_genes, max_iso=4)
        alpha_perturb = rng.gamma(
            shape=0.5, scale=2.0, size=(n_genes, 4)).astype(float) + 0.1
        cluster_props[cl] = alpha_perturb

    # ----- Build transcript count matrix ----------------------------------
    # Store as COO triplets; fill column-by-column (transcript)
    coo_rows_all = []
    coo_cols_all = []
    coo_vals_all = []

    for gi, gene in enumerate(gene_names):
        n_iso = int(n_isoforms_per_gene[gi])
        tx_col_indices = gene_to_tx_indices[gene]  # len == n_iso

        gene_counts = X_counts[:, gi]  # (n_cells,)

        # Build (n_cells, n_iso) proportion matrix
        props = np.zeros((n_cells, n_iso), dtype=float)
        for ci, cell_bc in enumerate(barcodes):
            cl = louvain[ci]
            alpha_gi = cluster_props[cl][gi, :n_iso]
            # Sample from Dirichlet with this cluster's alpha
            p = rng.dirichlet(alpha_gi)
            props[ci] = p

        # Distribute integer counts via multinomial per cell
        tx_counts = np.zeros((n_cells, n_iso), dtype=np.int32)
        for ci in range(n_cells):
            total = int(gene_counts[ci])
            if total == 0:
                continue
            tx_counts[ci] = rng.multinomial(total, props[ci])

        # Accumulate COO entries (features x cells convention for MEX)
        for iso_idx, tx_col in enumerate(tx_col_indices):
            col_vals = tx_counts[:, iso_idx]
            nonzero_cells = np.where(col_vals > 0)[0]
            for ci in nonzero_cells:
                # In MEX: rows=features(transcripts), cols=cells
                coo_rows_all.append(tx_col)
                coo_cols_all.append(ci)
                coo_vals_all.append(float(col_vals[ci]))

    logger.info("  Assembling sparse transcript matrix...")
    tx_matrix = scipy.sparse.csc_matrix(
        (coo_vals_all,
         (coo_rows_all, coo_cols_all)),
        shape=(n_transcripts, n_cells),
    )

    _write_mex(out_dir, tx_matrix, barcodes, transcript_ids)

    # ----- Write gene-transcript map TSV ----------------------------------
    map_out.parent.mkdir(parents=True, exist_ok=True)
    with open(map_out, "w") as fh:
        fh.write("transcript_id\tgene_id\n")
        for tx_id, gene_id in tx_to_gene.items():
            fh.write(f"{tx_id}\t{gene_id}\n")
    logger.info(f"  Gene-transcript map written to {map_out}")

    return transcript_ids, tx_to_gene


# ---------------------------------------------------------------------------
# Step 4: Run IRIS pipeline modules
# ---------------------------------------------------------------------------

class _Args:
    """Simple namespace mirroring argparse.Namespace."""
    pass


def _timed(name, func):
    """Run func(), return (result_or_None, elapsed_seconds)."""
    t0 = time.time()
    result = None
    error = None
    try:
        result = func()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.error(f"  [{name}] FAILED: {error}")
        logger.debug(traceback.format_exc())
    elapsed = round(time.time() - t0, 2)
    if error is None:
        logger.info(f"  [{name}] done in {elapsed}s")
    return result, elapsed, error


def run_dual_layer_clustering(
        gene_matrix_dir, transcript_matrix_dir, gt_map, out_dir):
    """Run dual_layer_clustering and return (success, elapsed, error)."""

    def _run():
        from iris.dual_layer_clustering import main as run_fn
        a = _Args()
        a.gene_matrix_dir = gene_matrix_dir
        a.transcript_matrix_dir = transcript_matrix_dir
        a.gene_transcript_map = gt_map
        a.output_gene_clusters = out_dir / "gene_clusters.tsv"
        a.output_isoform_clusters = out_dir / "isoform_clusters.tsv"
        a.output_joint_clusters = out_dir / "joint_clusters.tsv"
        a.output_joint_umap = out_dir / "joint.umap.tsv"
        a.output_diversity = out_dir / "isoform_diversity.tsv"
        a.output_comparison = out_dir / "cluster_comparison.json"
        a.cluster_method = "leiden"
        a.resolution = 0.5
        a.isoform_resolution = 0.5
        a.n_neighbors = 15
        a.n_pcs = 30
        a.min_isoforms_per_gene = 2
        a.diversity_metric = "shannon"
        a.isoform_weight = 3.0
        a.normalize_pcs = True
        a.random_state = 42
        run_fn(a)

    _, elapsed, error = _timed("dual_layer_clustering", _run)
    return error is None, elapsed, error


def run_cell_type_annotation(gene_matrix_dir, clusters_file, out_dir):
    """Run cell_type_annotation and return (success, elapsed, error)."""

    def _run():
        from iris.cell_type_annotation import main as run_fn
        a = _Args()
        a.gene_matrix_dir = gene_matrix_dir
        a.clusters = clusters_file
        a.marker_genes_db = None
        a.output_annotations = out_dir / "cell_type_annotations.tsv"
        a.output_cluster_types = out_dir / "cluster_cell_types.tsv"
        a.output_summary = out_dir / "cell_type_summary.json"
        a.method = "marker_overlap"
        a.min_marker_genes = 2  # matches default
        a.cluster_column = "cluster"
        a.species = "human"
        run_fn(a)

    _, elapsed, error = _timed("cell_type_annotation", _run)
    return error is None, elapsed, error


def run_differential_transcript_usage(
        transcript_matrix_dir, clusters_file, gt_map, out_dir):
    """Run DTU module and return (success, elapsed, error)."""

    def _run():
        from iris.differential_transcript_usage import main as run_fn
        a = _Args()
        a.transcript_matrix_dir = transcript_matrix_dir
        a.clusters = clusters_file
        a.gene_transcript_map = gt_map
        a.output_dtu = out_dir / "dtu_results.tsv"
        a.output_switching = out_dir / "isoform_switching.tsv"
        a.output_summary = out_dir / "dtu_summary.json"
        a.test_method = "chi_squared"
        a.min_cells_per_cluster = 10
        a.min_gene_counts = 10
        a.min_isoforms = 2
        a.fdr_threshold = 0.05
        a.cluster_column = "cluster"
        a.comparison_mode = "one_vs_rest"
        a.yates_correction = True
        a.dm_maxiter = 200
        a.n_jobs = 1
        run_fn(a)

    _, elapsed, error = _timed("differential_transcript_usage", _run)
    return error is None, elapsed, error


def run_isoform_trajectory(
        gene_matrix_dir, transcript_matrix_dir, gt_map,
        clusters_file, out_dir):
    """Run isoform_trajectory and return (success, elapsed, error)."""

    def _run():
        from iris.isoform_trajectory import main as run_fn
        a = _Args()
        a.gene_matrix_dir = gene_matrix_dir
        a.transcript_matrix_dir = transcript_matrix_dir
        a.gene_transcript_map = gt_map
        a.clusters = clusters_file
        a.output_pseudotime = out_dir / "pseudotime.tsv"
        a.output_isoform_dynamics = out_dir / "isoform_dynamics.tsv"
        a.output_switching_trajectory = out_dir / "trajectory_switching.tsv"
        a.output_summary = out_dir / "trajectory_summary.json"
        a.n_dpt_neighbors = 15
        a.n_pcs = 30
        a.min_isoforms = 2
        a.n_bins = 10
        a.random_state = 42
        a.n_jobs = 1
        run_fn(a)

    _, elapsed, error = _timed("isoform_trajectory", _run)
    return error is None, elapsed, error


def run_export(gene_matrix_dir, transcript_matrix_dir, out_dir):
    """Run export_anndata and return (success, elapsed, error)."""

    def _run():
        from iris.export_anndata import main as run_fn
        a = _Args()
        a.gene_matrix_dir = gene_matrix_dir
        a.transcript_matrix_dir = transcript_matrix_dir
        a.gene_clusters = out_dir / "gene_clusters.tsv"
        a.isoform_clusters = out_dir / "isoform_clusters.tsv"
        a.joint_clusters = out_dir / "joint_clusters.tsv"
        a.joint_umap = out_dir / "joint.umap.tsv"
        a.cell_type_annotations = out_dir / "cell_type_annotations.tsv"
        a.isoform_diversity = out_dir / "isoform_diversity.tsv"
        a.dtu_results = out_dir / "dtu_results.tsv"
        a.switching_results = out_dir / "isoform_switching.tsv"
        a.novel_catalog = None
        a.novel_enrichment = None
        a.cluster_comparison = out_dir / "cluster_comparison.json"
        a.pseudotime = out_dir / "pseudotime.tsv"
        a.ase_results = None
        a.output = out_dir / "iris.h5ad"
        run_fn(a)

    _, elapsed, error = _timed("export", _run)
    return error is None, elapsed, error


# ---------------------------------------------------------------------------
# Step 5: Compute validation metrics
# ---------------------------------------------------------------------------

def _load_cluster_tsv(path, cluster_col="cluster"):
    """Load a cluster TSV and return a barcode->cluster Series."""
    df = pd.read_csv(path, sep="\t")
    barcode_col = df.columns[0]
    return pd.Series(
        df[cluster_col].astype(str).values,
        index=df[barcode_col].values,
        name=cluster_col,
    )


def compute_clustering_metrics(adata, out_dir):
    """Compare IRIS cluster assignments against known Louvain labels.

    Returns a dict with ARI and NMI for gene, isoform and joint clusters.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    known_labels = adata.obs["louvain"].values
    barcodes = list(adata.obs_names)
    known_series = pd.Series(known_labels, index=barcodes)

    metrics = {}

    cluster_files = {
        "gene_clusters": out_dir / "gene_clusters.tsv",
        "isoform_clusters": out_dir / "isoform_clusters.tsv",
        "joint_clusters": out_dir / "joint_clusters.tsv",
    }

    for name, path in cluster_files.items():
        if not path.exists():
            logger.warning(f"  {name}: file not found, skipping metrics.")
            metrics[name] = {"ari": None, "nmi": None, "n_clusters": None}
            continue

        iris_labels = _load_cluster_tsv(path)
        # Align on common barcodes
        common = sorted(set(barcodes) & set(iris_labels.index))
        if len(common) == 0:
            logger.warning(f"  {name}: no overlapping barcodes.")
            metrics[name] = {"ari": None, "nmi": None, "n_clusters": None}
            continue

        y_true = known_series.loc[common].values
        y_pred = iris_labels.loc[common].values

        ari = float(adjusted_rand_score(y_true, y_pred))
        nmi = float(normalized_mutual_info_score(
            y_true, y_pred, average_method="arithmetic"))
        n_clusters = int(pd.Series(y_pred).nunique())

        metrics[name] = {
            "ari": round(ari, 4),
            "nmi": round(nmi, 4),
            "n_clusters": n_clusters,
            "n_cells_evaluated": len(common),
        }
        logger.info(
            f"  {name}: ARI={ari:.4f}  NMI={nmi:.4f}  "
            f"n_clusters={n_clusters}  n_cells={len(common)}")

    return metrics


def compute_annotation_accuracy(adata, out_dir):
    """Compare IRIS cell type annotations against known Louvain types.

    The PBMC 3k Louvain labels map onto canonical PBMC cell types.
    We build a simple ground-truth map (Louvain cluster -> broad cell type),
    then check what fraction of IRIS-annotated clusters agree.

    Returns a dict with accuracy metrics.
    """
    # Known PBMC 3k Louvain -> broad cell type mapping
    # (from the Scanpy PBMC 3k tutorial canonical annotations)
    LOUVAIN_TO_TYPE = {
        "CD4 T cells": "T cells",
        "CD14+ Monocytes": "Monocytes",
        "B cells": "B cells",
        "CD8 T cells": "T cells",
        "NK cells": "NK cells",
        "FCGR3A+ Monocytes": "Monocytes",
        "Dendritic cells": "Dendritic cells",
        "Megakaryocytes": "Platelets",
    }

    ann_path = out_dir / "cluster_cell_types.tsv"
    if not ann_path.exists():
        logger.warning("  cell_type_annotations: file not found.")
        return {"accuracy": None, "n_clusters_evaluated": 0}

    cluster_types = pd.read_csv(ann_path, sep="\t")
    # column: cluster (IRIS cluster id) -> cell_type (IRIS predicted)

    # Map known louvain labels to broad types for each cell, then
    # determine the dominant Louvain label per IRIS cluster to compare.
    joint_clusters_path = out_dir / "joint_clusters.tsv"
    if not joint_clusters_path.exists():
        joint_clusters_path = out_dir / "gene_clusters.tsv"
    if not joint_clusters_path.exists():
        return {"accuracy": None, "n_clusters_evaluated": 0}

    iris_clusters = _load_cluster_tsv(joint_clusters_path)
    barcodes = list(adata.obs_names)
    louvain = adata.obs["louvain"]

    common = sorted(set(barcodes) & set(iris_clusters.index))
    if len(common) == 0:
        return {"accuracy": None, "n_clusters_evaluated": 0}

    # Build per-IRIS-cluster dominant broad type
    df = pd.DataFrame({
        "iris_cluster": iris_clusters.loc[common].values,
        "louvain": louvain.loc[common].values,
    })
    df["broad_type"] = df["louvain"].map(LOUVAIN_TO_TYPE).fillna("Unknown")

    cluster_dominant = (
        df.groupby("iris_cluster")["broad_type"]
        .agg(lambda x: x.value_counts().index[0])
    )

    # Compare against IRIS annotation
    type_map = dict(zip(
        cluster_types["cluster"].astype(str),
        cluster_types["cell_type"],
    ))

    n_correct = 0
    n_total = 0
    per_cluster = []
    for cl, true_type in cluster_dominant.items():
        iris_type = type_map.get(str(cl), "Unknown")
        # Strict accuracy: only count non-Unknown predicted clusters
        if iris_type != "Unknown":
            n_total += 1
            if true_type.lower() in iris_type.lower() or \
                    iris_type.lower() in true_type.lower():
                n_correct += 1
        per_cluster.append({
            "iris_cluster": cl,
            "dominant_true_type": true_type,
            "iris_predicted_type": iris_type,
            "correct": (true_type.lower() in iris_type.lower()
                        or iris_type.lower() in true_type.lower()),
        })

    accuracy = round(n_correct / n_total, 4) if n_total > 0 else None
    if accuracy is not None:
        logger.info(
            f"  Cell type annotation accuracy: "
            f"{n_correct}/{n_total} annotated clusters correct "
            f"({accuracy * 100:.1f}%)")
    else:
        logger.info(
            "  Cell type annotation accuracy: N/A (no annotated clusters)")

    return {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_annotated_clusters": n_total,
        "n_clusters_evaluated": len(cluster_dominant),
        "per_cluster_detail": per_cluster,
    }


def collect_dtu_summary(out_dir):
    """Parse DTU summary JSON and return key metrics."""
    path = out_dir / "dtu_summary.json"
    if not path.exists():
        return {"n_significant": None, "n_switching": None}
    try:
        with open(path) as fh:
            data = json.load(fh)
        return {
            "n_significant": data.get("n_significant_genes"),
            "n_switching": data.get("n_switching_events"),
            "n_genes_tested": data.get("n_genes_tested"),
            "fdr_threshold": data.get("fdr_threshold"),
        }
    except Exception as exc:
        logger.warning(f"  Could not parse DTU summary: {exc}")
        return {"n_significant": None, "n_switching": None}


# ---------------------------------------------------------------------------
# Step 6: Validation report
# ---------------------------------------------------------------------------

def build_report(
        iris_version,
        clustering_metrics,
        annotation_metrics,
        dtu_summary,
        step_runtimes,
        step_errors,
        n_cells,
        n_genes,
        n_transcripts,
):
    """Assemble the JSON validation report."""
    report = {
        "iris_version": iris_version,
        "dataset": "pbmc3k_processed",
        "n_cells": n_cells,
        "n_genes": n_genes,
        "n_synthetic_transcripts": n_transcripts,
        "clustering_metrics": clustering_metrics,
        "cell_type_annotation": annotation_metrics,
        "dtu_summary": dtu_summary,
        "step_runtimes_s": step_runtimes,
        "step_errors": step_errors,
    }
    return report


# ---------------------------------------------------------------------------
# Step 7: Print summary table
# ---------------------------------------------------------------------------

def print_summary(report):
    """Print a human-readable summary table to stdout."""

    def _row(label, value, width=38):
        label_str = f"  {label}"
        value_str = str(value) if value is not None else "N/A"
        print(f"{label_str:<{width}} {value_str}")

    sep = "=" * 60
    print(sep)
    print("  IRIS PBMC 3k Validation Summary")
    print(sep)
    _row("IRIS version", report["iris_version"])
    _row("Dataset", report["dataset"])
    _row("N cells", report["n_cells"])
    _row("N genes", report["n_genes"])
    _row("N synthetic transcripts", report["n_synthetic_transcripts"])
    print()

    print("  -- Clustering quality (vs. known Louvain labels) --")
    for layer_name, m in report["clustering_metrics"].items():
        if m["ari"] is not None:
            _row(
                f"{layer_name} ARI",
                f"{m['ari']:.4f}  (n_clusters={m['n_clusters']})")
            _row(f"{layer_name} NMI", f"{m['nmi']:.4f}")
        else:
            _row(f"{layer_name}", "FAILED / not computed")
    print()

    print("  -- Cell type annotation --")
    ann = report["cell_type_annotation"]
    _row("Annotated clusters correct", ann.get("n_correct"))
    _row("Total annotated clusters", ann.get("n_annotated_clusters"))
    _row("Annotation accuracy", ann.get("accuracy"))
    print()

    print("  -- DTU summary --")
    dtu = report["dtu_summary"]
    _row("N genes tested", dtu.get("n_genes_tested"))
    _row("N significant (FDR<0.05)", dtu.get("n_significant"))
    _row("N isoform switching events", dtu.get("n_switching"))
    print()

    print("  -- Step runtimes (seconds) --")
    for step, rt in report["step_runtimes_s"].items():
        status = "  [FAILED]" if report["step_errors"].get(step) else ""
        _row(step, f"{rt}{status}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gene_matrix_dir = out_dir / "gene_matrix"
    transcript_matrix_dir = out_dir / "transcript_matrix"
    gt_map = out_dir / "gene_transcript_map.tsv"
    iris_out = out_dir / "iris_output"
    iris_out.mkdir(parents=True, exist_ok=True)

    step_runtimes = {}
    step_errors = {}
    n_transcripts = 0

    # ------------------------------------------------------------------
    # 1. Download PBMC 3k
    # ------------------------------------------------------------------
    t0 = time.time()
    adata = download_pbmc3k()
    step_runtimes["download_pbmc3k"] = round(time.time() - t0, 2)
    step_errors["download_pbmc3k"] = None

    n_cells = int(adata.shape[0])
    n_genes = int(adata.shape[1])

    # ------------------------------------------------------------------
    # 2. Save gene MEX matrix
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        barcodes, gene_names = save_gene_matrix(adata, gene_matrix_dir)
        step_errors["save_gene_matrix"] = None
    except Exception as exc:
        logger.error(f"save_gene_matrix FAILED: {exc}")
        step_errors["save_gene_matrix"] = str(exc)
        sys.exit(1)
    step_runtimes["save_gene_matrix"] = round(time.time() - t0, 2)

    # ------------------------------------------------------------------
    # 3. Generate synthetic transcript data
    # ------------------------------------------------------------------
    t0 = time.time()
    try:
        transcript_ids, tx_to_gene = generate_transcript_data(
            adata, barcodes, gene_names,
            transcript_matrix_dir, gt_map, seed=args.seed)
        n_transcripts = len(transcript_ids)
        step_errors["generate_transcripts"] = None
    except Exception as exc:
        logger.error(f"generate_transcript_data FAILED: {exc}")
        step_errors["generate_transcripts"] = str(exc)
        sys.exit(1)
    step_runtimes["generate_transcripts"] = round(time.time() - t0, 2)

    # ------------------------------------------------------------------
    # 4a. Dual-layer clustering
    # ------------------------------------------------------------------
    logger.info("Running IRIS dual_layer_clustering...")
    ok, elapsed, err = run_dual_layer_clustering(
        gene_matrix_dir, transcript_matrix_dir, gt_map, iris_out)
    step_runtimes["dual_layer_clustering"] = elapsed
    step_errors["dual_layer_clustering"] = err

    # Determine which cluster file to pass downstream
    clusters_file = iris_out / "joint_clusters.tsv"
    if not clusters_file.exists():
        clusters_file = iris_out / "gene_clusters.tsv"

    # ------------------------------------------------------------------
    # 4b. Cell type annotation
    # ------------------------------------------------------------------
    logger.info("Running IRIS cell_type_annotation...")
    ok_ann, elapsed_ann, err_ann = run_cell_type_annotation(
        gene_matrix_dir, clusters_file, iris_out)
    step_runtimes["cell_type_annotation"] = elapsed_ann
    step_errors["cell_type_annotation"] = err_ann

    # ------------------------------------------------------------------
    # 4c. Differential transcript usage
    # ------------------------------------------------------------------
    logger.info("Running IRIS differential_transcript_usage...")
    ok_dtu, elapsed_dtu, err_dtu = run_differential_transcript_usage(
        transcript_matrix_dir, clusters_file, gt_map, iris_out)
    step_runtimes["differential_transcript_usage"] = elapsed_dtu
    step_errors["differential_transcript_usage"] = err_dtu

    # ------------------------------------------------------------------
    # 4d. Isoform trajectory
    # ------------------------------------------------------------------
    logger.info("Running IRIS isoform_trajectory...")
    ok_traj, elapsed_traj, err_traj = run_isoform_trajectory(
        gene_matrix_dir, transcript_matrix_dir, gt_map,
        clusters_file, iris_out)
    step_runtimes["isoform_trajectory"] = elapsed_traj
    step_errors["isoform_trajectory"] = err_traj

    # ------------------------------------------------------------------
    # 4e. Export
    # ------------------------------------------------------------------
    logger.info("Running IRIS export...")
    ok_exp, elapsed_exp, err_exp = run_export(
        gene_matrix_dir, transcript_matrix_dir, iris_out)
    step_runtimes["export"] = elapsed_exp
    step_errors["export"] = err_exp

    # ------------------------------------------------------------------
    # 5. Compute validation metrics
    # ------------------------------------------------------------------
    logger.info("Computing validation metrics...")

    clustering_metrics = compute_clustering_metrics(adata, iris_out)
    annotation_metrics = compute_annotation_accuracy(adata, iris_out)
    dtu_summary = collect_dtu_summary(iris_out)

    # ------------------------------------------------------------------
    # 6. Build and save report
    # ------------------------------------------------------------------
    try:
        import iris
        iris_version = iris.__version__
    except Exception:
        iris_version = "unknown"

    report = build_report(
        iris_version=iris_version,
        clustering_metrics=clustering_metrics,
        annotation_metrics=annotation_metrics,
        dtu_summary=dtu_summary,
        step_runtimes=step_runtimes,
        step_errors=step_errors,
        n_cells=n_cells,
        n_genes=n_genes,
        n_transcripts=n_transcripts,
    )

    report_path = out_dir / "validation_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info(f"Validation report written to {report_path}")

    # ------------------------------------------------------------------
    # 7. Print summary table
    # ------------------------------------------------------------------
    print_summary(report)

    # Exit non-zero if any core IRIS step failed
    core_steps = [
        "dual_layer_clustering",
        "cell_type_annotation",
        "differential_transcript_usage",
    ]
    failed = [s for s in core_steps if step_errors.get(s)]
    if failed:
        logger.warning(
            f"Validation completed with failures in: {failed}")
        sys.exit(1)

    logger.info("Validation complete.")


if __name__ == "__main__":
    main()
