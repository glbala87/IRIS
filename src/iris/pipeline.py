"""IRIS full pipeline orchestrator with checkpointing and error recovery.

Runs all IRIS analysis modules in sequence on pre-processed
single-cell long-read data. Supports:
  - Checkpointing: tracks completed steps, resumes from last failure
  - Error recovery: failing modules don't crash the pipeline
  - File logging: all output goes to iris.log in the output directory
  - Input validation: checks inputs before starting

Inputs:
  - Gene expression MEX matrix directory (required)
  - Transcript expression MEX matrix directory (required)
  - Gene-transcript mapping TSV (required)
  - Tagged BAM file (optional, for ASE)
  - Gffcompare annotated GTFs (optional, for novel isoform discovery)
  - VCF file (optional, for ASE)
"""
import json
import logging
import sys
import time
import traceback
from pathlib import Path

from ._logging import get_named_logger, wf_parser


STEPS = [
    'clustering', 'dual_clustering', 'annotation',
    'dtu', 'novel_isoforms', 'trajectory', 'ase', 'export',
]


def argparser():
    """Create argument parser for the full pipeline."""
    parser = wf_parser("run")

    # Required inputs
    parser.add_argument(
        "gene_matrix_dir", type=Path,
        help="Gene expression MEX matrix directory.")
    parser.add_argument(
        "transcript_matrix_dir", type=Path,
        help="Transcript expression MEX matrix directory.")
    parser.add_argument(
        "--gene_transcript_map", type=Path, required=True,
        help="TSV mapping transcript_id to gene_id.")

    # Optional inputs
    parser.add_argument(
        "--tagged_bam", type=Path, default=None,
        help="Tagged BAM file with CB tags (for ASE).")
    parser.add_argument(
        "--annotated_gtfs", type=Path, nargs='*', default=None,
        help="Gffcompare annotated GTF files (for novel isoforms).")
    parser.add_argument(
        "--vcf", type=Path, default=None,
        help="VCF with known variants (for ASE).")
    parser.add_argument(
        "--marker_genes_db", type=Path, default=None,
        help="Custom marker gene database TSV.")

    # Output
    parser.add_argument(
        "--out_dir", type=Path, default=Path("iris_results"),
        help="Output directory.")
    parser.add_argument(
        "--log_file", type=str, default=None,
        help="Log file path. Default: <out_dir>/iris.log")

    # Module toggles
    parser.add_argument(
        "--skip_clustering", action='store_true',
        help="Skip basic clustering.")
    parser.add_argument(
        "--skip_dual_clustering", action='store_true',
        help="Skip dual-layer clustering.")
    parser.add_argument(
        "--skip_dtu", action='store_true',
        help="Skip differential transcript usage.")
    parser.add_argument(
        "--skip_novel", action='store_true',
        help="Skip novel isoform discovery.")
    parser.add_argument(
        "--skip_trajectory", action='store_true',
        help="Skip isoform trajectory analysis.")
    parser.add_argument(
        "--skip_ase", action='store_true',
        help="Skip allele-specific expression.")
    parser.add_argument(
        "--skip_annotation", action='store_true',
        help="Skip cell type annotation.")

    # Checkpointing
    parser.add_argument(
        "--resume", action='store_true',
        help="Resume from last checkpoint (skip completed steps).")
    parser.add_argument(
        "--force", action='store_true',
        help="Re-run all steps even if outputs exist.")

    # Common params
    parser.add_argument(
        "--cluster_method", type=str, default="leiden",
        choices=["leiden", "louvain"])
    parser.add_argument(
        "--cluster_resolution", type=float, default=1.0)
    parser.add_argument(
        "--species", type=str, default="human",
        choices=["human", "mouse"])
    parser.add_argument(
        "--n_jobs", type=int, default=1,
        help="Number of parallel worker processes for DTU, ASE, and "
             "trajectory modules. Use -1 to use all available CPUs.")
    parser.add_argument(
        "--max_dense_gb", type=float, default=2.0,
        help="Maximum memory (GB) allowed for dense matrix operations "             "(default: 2.0).")
    parser.add_argument(
        "--random_state", type=int, default=0,
        help="Random seed for reproducibility across all modules.")

    return parser


class Checkpoint:
    """Track pipeline progress for resume capability."""

    def __init__(self, out_dir):
        self.path = Path(out_dir) / ".iris_checkpoint.json"
        self.state = self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path) as fh:
                return json.load(fh)
        return {'completed': [], 'failed': [], 'skipped': []}

    def _save(self):
        with open(self.path, 'w') as fh:
            json.dump(self.state, fh, indent=2)

    def is_done(self, step):
        return step in self.state['completed']

    def mark_done(self, step, duration_s=None):
        if step not in self.state['completed']:
            self.state['completed'].append(step)
        entry = {'step': step, 'duration_s': duration_s}
        self.state[f'{step}_info'] = entry
        self._save()

    def mark_failed(self, step, error_msg):
        if step not in self.state['failed']:
            self.state['failed'].append(step)
        self.state[f'{step}_error'] = error_msg
        self._save()

    def mark_skipped(self, step, reason):
        if step not in self.state['skipped']:
            self.state['skipped'].append(step)
        self.state[f'{step}_skip_reason'] = reason
        self._save()

    def reset(self):
        self.state = {'completed': [], 'failed': [], 'skipped': []}
        self._save()


def _setup_file_logging(log_file):
    """Add file handler to root logger."""
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '[%(asctime)s - %(name)s] %(message)s', datefmt='%H:%M:%S'))
    logging.getLogger().addHandler(file_handler)


def _run_step(name, func, checkpoint, logger, force=False):
    """Run a pipeline step with error handling and checkpointing.

    Returns True if step completed, False if it failed.
    """
    if checkpoint.is_done(name) and not force:
        logger.info(f"  [{name}] Already complete (resume mode).")
        return True

    start = time.time()
    try:
        func()
        duration = time.time() - start
        checkpoint.mark_done(name, duration_s=round(duration, 1))
        logger.info(
            f"  [{name}] Complete ({duration:.1f}s).")
        return True
    except Exception as e:
        duration = time.time() - start
        error_msg = f"{type(e).__name__}: {e}"
        checkpoint.mark_failed(name, error_msg)
        logger.error(
            f"  [{name}] FAILED after {duration:.1f}s: {error_msg}")
        logger.debug(traceback.format_exc())
        return False


class _Args:
    """Simple namespace for passing args to modules."""
    pass


def main(args):
    """Run the full IRIS pipeline with checkpointing."""
    logger = get_named_logger("Pipeline")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    log_file = args.log_file or str(out / "iris.log")
    _setup_file_logging(log_file)

    # Apply configurable memory limit for dense matrix operations
    from ._stats import set_max_dense_bytes
    set_max_dense_bytes(int(args.max_dense_gb * 1024 ** 3))

    logger.info("=" * 60)
    logger.info("IRIS pipeline starting")
    logger.info("=" * 60)
    logger.info(f"  Gene matrix:       {args.gene_matrix_dir}")
    logger.info(f"  Transcript matrix: {args.transcript_matrix_dir}")
    logger.info(f"  Output:            {out}")
    logger.info(f"  Log file:          {log_file}")

    # Input validation
    try:
        from .validate import validate_pipeline_inputs
        validate_pipeline_inputs(args)
        logger.info("  Input validation: PASSED")
    except ImportError:
        logger.info("  Input validation: skipped (module not available)")
    except Exception as e:
        logger.error(f"  Input validation FAILED: {e}")
        logger.error("  Fix the issues above and re-run.")
        sys.exit(1)

    # Checkpointing
    checkpoint = Checkpoint(out)
    if args.force:
        checkpoint.reset()
        logger.info("  Force mode: re-running all steps.")
    elif args.resume:
        done = checkpoint.state.get('completed', [])
        if done:
            logger.info(
                f"  Resume mode: {len(done)} steps already done: "
                f"{', '.join(done)}")

    clusters_file = None
    results = {}
    n_success = 0
    n_failed = 0
    n_skipped = 0

    # --- 1. Basic clustering ---
    if args.skip_clustering:
        checkpoint.mark_skipped('clustering', 'user skipped')
        n_skipped += 1
    else:
        def _run_clustering():
            from .cluster_analysis import main as run_fn
            a = _Args()
            a.matrix_dir = args.gene_matrix_dir
            a.output_clusters = out / "clusters.tsv"
            a.output_umap = out / "cluster.umap.tsv"
            a.output_markers = out / "marker_genes.tsv"
            a.output_summary = out / "cluster_summary.json"
            a.cluster_method = args.cluster_method
            a.resolution = args.cluster_resolution
            a.n_neighbors = 15
            a.n_pcs = 50
            a.n_marker_genes = 25
            a.marker_method = "wilcoxon"
            a.normalization = "scanpy"
            a.norm_count = 10000
            a.cellranger_cell_calling = False
            a.expected_cells = 3000
            a.min_dist = 0.3
            a.spread = 1.0
            a.random_state = args.random_state
            run_fn(a)

        if _run_step(
                'clustering', _run_clustering,
                checkpoint, logger, args.force):
            clusters_file = out / "clusters.tsv"
            results['clusters'] = str(clusters_file)
            n_success += 1
        else:
            n_failed += 1

    # --- 2. Dual-layer clustering ---
    if args.skip_dual_clustering:
        checkpoint.mark_skipped('dual_clustering', 'user skipped')
        n_skipped += 1
    else:
        def _run_dual():
            from .dual_layer_clustering import main as run_fn
            a = _Args()
            a.gene_matrix_dir = args.gene_matrix_dir
            a.transcript_matrix_dir = args.transcript_matrix_dir
            a.gene_transcript_map = args.gene_transcript_map
            a.output_gene_clusters = out / "gene_clusters.tsv"
            a.output_isoform_clusters = out / "isoform_clusters.tsv"
            a.output_joint_clusters = out / "joint_clusters.tsv"
            a.output_joint_umap = out / "joint.umap.tsv"
            a.output_diversity = out / "isoform_diversity.tsv"
            a.output_comparison = out / "cluster_comparison.json"
            a.cluster_method = args.cluster_method
            a.resolution = args.cluster_resolution
            a.isoform_resolution = args.cluster_resolution
            a.n_neighbors = 15
            a.n_pcs = 50
            a.min_isoforms_per_gene = 2
            a.diversity_metric = "shannon"
            a.isoform_weight = 3.0
            a.normalize_pcs = True
            a.random_state = args.random_state
            run_fn(a)

        if _run_step(
                'dual_clustering', _run_dual,
                checkpoint, logger, args.force):
            clusters_file = out / "joint_clusters.tsv"
            results['dual_clustering'] = str(clusters_file)
            n_success += 1
        else:
            n_failed += 1

    # If we have checkpoint data for clusters but didn't run them
    if clusters_file is None and (out / "joint_clusters.tsv").exists():
        clusters_file = out / "joint_clusters.tsv"
    if clusters_file is None and (out / "clusters.tsv").exists():
        clusters_file = out / "clusters.tsv"

    if clusters_file is None or not clusters_file.exists():
        logger.warning(
            "No cluster assignments available. "
            "Skipping downstream modules.")
    else:
        # --- 3. Cell type annotation ---
        if args.skip_annotation:
            checkpoint.mark_skipped('annotation', 'user skipped')
            n_skipped += 1
        else:
            def _run_annotate():
                from .cell_type_annotation import main as run_fn
                a = _Args()
                a.gene_matrix_dir = args.gene_matrix_dir
                a.clusters = clusters_file
                a.marker_genes_db = args.marker_genes_db
                a.output_annotations = out / "cell_type_annotations.tsv"
                a.output_cluster_types = out / "cluster_cell_types.tsv"
                a.output_summary = out / "cell_type_summary.json"
                a.method = "marker_overlap"
                a.min_marker_genes = 3
                a.cluster_column = "cluster"
                a.species = args.species
                run_fn(a)

            if _run_step(
                    'annotation', _run_annotate,
                    checkpoint, logger, args.force):
                results['cell_types'] = str(
                    out / "cluster_cell_types.tsv")
                n_success += 1
            else:
                n_failed += 1

        # --- 4. DTU ---
        if args.skip_dtu:
            checkpoint.mark_skipped('dtu', 'user skipped')
            n_skipped += 1
        else:
            def _run_dtu():
                from .differential_transcript_usage import \
                    main as run_fn
                a = _Args()
                a.transcript_matrix_dir = args.transcript_matrix_dir
                a.clusters = clusters_file
                a.gene_transcript_map = args.gene_transcript_map
                a.output_dtu = out / "dtu_results.tsv"
                a.output_switching = out / "isoform_switching.tsv"
                a.output_summary = out / "dtu_summary.json"
                a.test_method = "chi_squared"
                a.min_cells_per_cluster = 10
                a.min_gene_counts = 20
                a.min_isoforms = 2
                a.fdr_threshold = 0.05
                a.cluster_column = "cluster"
                a.n_jobs = args.n_jobs
                run_fn(a)

            if _run_step(
                    'dtu', _run_dtu,
                    checkpoint, logger, args.force):
                results['dtu'] = str(out / "dtu_results.tsv")
                n_success += 1
            else:
                n_failed += 1

        # --- 5. Novel isoforms ---
        if args.skip_novel:
            checkpoint.mark_skipped('novel_isoforms', 'user skipped')
            n_skipped += 1
        elif not args.annotated_gtfs:
            checkpoint.mark_skipped(
                'novel_isoforms', 'no annotated GTFs provided')
            n_skipped += 1
        else:
            def _run_novel():
                from .novel_isoform_discovery import main as run_fn
                a = _Args()
                a.transcript_matrix_dir = args.transcript_matrix_dir
                a.annotated_gtfs = args.annotated_gtfs
                a.clusters = clusters_file
                a.gene_transcript_map = args.gene_transcript_map
                a.output_novel_catalog = \
                    out / "novel_isoform_catalog.tsv"
                a.output_cluster_enrichment = \
                    out / "novel_isoform_enrichment.tsv"
                a.output_summary = \
                    out / "novel_isoform_summary.json"
                a.min_cells = 3
                a.min_counts = 5
                a.enrichment_fdr = 0.05
                a.cluster_column = "cluster"
                run_fn(a)

            if _run_step(
                    'novel_isoforms', _run_novel,
                    checkpoint, logger, args.force):
                results['novel'] = str(
                    out / "novel_isoform_catalog.tsv")
                n_success += 1
            else:
                n_failed += 1

        # --- 6. Trajectory ---
        if args.skip_trajectory:
            checkpoint.mark_skipped('trajectory', 'user skipped')
            n_skipped += 1
        else:
            def _run_trajectory():
                from .isoform_trajectory import main as run_fn
                a = _Args()
                a.gene_matrix_dir = args.gene_matrix_dir
                a.transcript_matrix_dir = args.transcript_matrix_dir
                a.gene_transcript_map = args.gene_transcript_map
                a.clusters = clusters_file
                a.output_pseudotime = out / "pseudotime.tsv"
                a.output_isoform_dynamics = \
                    out / "isoform_dynamics.tsv"
                a.output_switching_trajectory = \
                    out / "trajectory_switching.tsv"
                a.output_summary = out / "trajectory_summary.json"
                a.n_dpt_neighbors = 15
                a.n_pcs = 30
                a.min_isoforms = 2
                a.n_bins = 10
                a.n_jobs = args.n_jobs
                a.random_state = args.random_state
                run_fn(a)

            if _run_step(
                    'trajectory', _run_trajectory,
                    checkpoint, logger, args.force):
                results['trajectory'] = str(out / "pseudotime.tsv")
                n_success += 1
            else:
                n_failed += 1

        # --- 7. ASE ---
        if args.skip_ase:
            checkpoint.mark_skipped('ase', 'user skipped')
            n_skipped += 1
        elif not args.tagged_bam:
            checkpoint.mark_skipped('ase', 'no tagged BAM provided')
            n_skipped += 1
        else:
            def _run_ase():
                from .allele_specific_expression import \
                    main as run_fn
                a = _Args()
                a.tagged_bam = args.tagged_bam
                a.vcf = args.vcf
                a.clusters = clusters_file
                a.output_ase = out / "ase_results.tsv"
                a.output_summary = out / "ase_summary.json"
                a.min_total_counts = 10
                a.min_cells = 5
                a.fdr_threshold = 0.05
                a.min_base_quality = 20
                a.n_jobs = args.n_jobs
                run_fn(a)

            if _run_step(
                    'ase', _run_ase,
                    checkpoint, logger, args.force):
                results['ase'] = str(out / "ase_results.tsv")
                n_success += 1
            else:
                n_failed += 1

    # --- 8. Export (always runs, collects whatever is available) ---
    def _run_export():
        from .export_anndata import main as run_fn
        a = _Args()
        a.gene_matrix_dir = args.gene_matrix_dir
        a.transcript_matrix_dir = args.transcript_matrix_dir
        a.gene_clusters = out / "gene_clusters.tsv"
        a.isoform_clusters = out / "isoform_clusters.tsv"
        a.joint_clusters = out / "joint_clusters.tsv"
        a.joint_umap = out / "joint.umap.tsv"
        a.cell_type_annotations = out / "cell_type_annotations.tsv"
        a.isoform_diversity = out / "isoform_diversity.tsv"
        a.dtu_results = out / "dtu_results.tsv"
        a.switching_results = out / "isoform_switching.tsv"
        a.novel_catalog = out / "novel_isoform_catalog.tsv"
        a.novel_enrichment = out / "novel_isoform_enrichment.tsv"
        a.cluster_comparison = out / "cluster_comparison.json"
        a.pseudotime = out / "pseudotime.tsv"
        a.ase_results = out / "ase_results.tsv"
        a.output = out / "iris.h5ad"
        run_fn(a)

    if _run_step('export', _run_export, checkpoint, logger, args.force):
        results['h5ad'] = str(out / "iris.h5ad")
        n_success += 1
    else:
        n_failed += 1

    # --- Summary ---
    summary = {
        'version': '0.1.0',
        'n_success': n_success,
        'n_failed': n_failed,
        'n_skipped': n_skipped,
        'completed_steps': checkpoint.state.get('completed', []),
        'failed_steps': checkpoint.state.get('failed', []),
        'skipped_steps': checkpoint.state.get('skipped', []),
        'outputs': results,
    }
    with open(out / "iris_pipeline_summary.json", 'w') as fh:
        json.dump(summary, fh, indent=2)

    logger.info("=" * 60)
    logger.info("IRIS pipeline finished")
    logger.info(f"  Success: {n_success}  Failed: {n_failed}  "
                f"Skipped: {n_skipped}")
    if n_failed > 0:
        failed = checkpoint.state.get('failed', [])
        logger.warning(f"  Failed steps: {', '.join(failed)}")
        logger.warning(f"  Fix issues and re-run with --resume")
    logger.info(f"  Results: {out}/")
    if (out / "iris.h5ad").exists():
        logger.info(
            f"  Load: adata = sc.read_h5ad('{out}/iris.h5ad')")
    logger.info(f"  Full log: {log_file}")
    logger.info("=" * 60)
