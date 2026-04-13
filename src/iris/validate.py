"""Input validation for the IRIS pipeline.

Validates all inputs before analysis begins, catching problems early
with clear error messages. Can be run standalone via `iris validate`
or called programmatically from the pipeline orchestrator.
"""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from ._logging import get_named_logger, wf_parser


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def argparser():
    """Create argument parser."""
    parser = wf_parser("validate")

    parser.add_argument(
        "--gene_matrix_dir", type=Path, required=True,
        help="Gene expression MEX matrix directory.")
    parser.add_argument(
        "--transcript_matrix_dir", type=Path, default=None,
        help="Transcript expression MEX matrix directory.")
    parser.add_argument(
        "--gene_transcript_map", type=Path, default=None,
        help="Gene-transcript mapping TSV.")
    parser.add_argument(
        "--tagged_bam", type=Path, default=None,
        help="Tagged BAM file.")
    parser.add_argument(
        "--annotated_gtfs", type=Path, nargs='*', default=None,
        help="Gffcompare annotated GTF files.")

    return parser


def _count_lines_gz(path):
    """Count lines in a gzipped file."""
    count = 0
    with gzip.open(path, 'rt') as fh:
        for _ in fh:
            count += 1
    return count


def _count_lines(path):
    """Count lines in a plain text file."""
    count = 0
    with open(path) as fh:
        for _ in fh:
            count += 1
    return count


def validate_mex_directory(path, name="matrix"):
    """Validate a MEX format matrix directory.

    Checks for required files and reads dimensions.

    :param path: Path to MEX directory.
    :param name: display name for error messages.
    :returns: dict with validation results.
    :raises ValidationError: if critical issues found.
    """
    logger = get_named_logger("Validate")
    path = Path(path)
    errors = []
    warnings = []

    if not path.exists():
        raise ValidationError(
            f"{name}: directory does not exist: {path}")
    if not path.is_dir():
        raise ValidationError(
            f"{name}: path is not a directory: {path}")

    # Check for required files
    mtx_candidates = ['matrix.mtx.gz', 'matrix.mtx']
    bc_candidates = ['barcodes.tsv.gz', 'barcodes.tsv']
    feat_candidates = ['features.tsv.gz', 'features.tsv',
                       'genes.tsv.gz', 'genes.tsv']

    mtx = next((path / f for f in mtx_candidates
                if (path / f).exists()), None)
    bc = next((path / f for f in bc_candidates
               if (path / f).exists()), None)
    feat = next((path / f for f in feat_candidates
                 if (path / f).exists()), None)

    if mtx is None:
        errors.append(f"Missing matrix file (matrix.mtx.gz)")
    if bc is None:
        errors.append(f"Missing barcodes file (barcodes.tsv.gz)")
    if feat is None:
        errors.append(f"Missing features file (features.tsv.gz)")

    if errors:
        raise ValidationError(
            f"{name} validation failed:\n  " +
            "\n  ".join(errors))

    # Count dimensions
    n_barcodes = 0
    n_features = 0
    try:
        counter = _count_lines_gz if str(bc).endswith('.gz') \
            else _count_lines
        n_barcodes = counter(bc)
    except Exception as e:
        errors.append(f"Cannot read barcodes: {e}")

    try:
        counter = _count_lines_gz if str(feat).endswith('.gz') \
            else _count_lines
        n_features = counter(feat)
    except Exception as e:
        errors.append(f"Cannot read features: {e}")

    if n_barcodes == 0:
        errors.append("Barcodes file is empty (0 cells)")
    if n_features == 0:
        errors.append("Features file is empty (0 features)")

    if n_barcodes < 10:
        warnings.append(
            f"Very few barcodes ({n_barcodes}) — check input")
    if n_features < 10:
        warnings.append(
            f"Very few features ({n_features}) — check input")

    if errors:
        raise ValidationError(
            f"{name} validation failed:\n  " +
            "\n  ".join(errors))

    logger.info(
        f"  {name}: {n_barcodes} cells x {n_features} features")

    return {
        'path': str(path),
        'n_barcodes': n_barcodes,
        'n_features': n_features,
        'valid': True,
        'warnings': warnings,
    }


def validate_tsv_file(path, required_columns=None, name="file"):
    """Validate a TSV file.

    :param path: Path to TSV.
    :param required_columns: list of required column names.
    :param name: display name.
    :returns: dict with validation results.
    :raises ValidationError: if critical issues found.
    """
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"{name}: file not found: {path}")

    try:
        df = pd.read_csv(path, sep='\t', nrows=5)
    except Exception as e:
        raise ValidationError(
            f"{name}: cannot parse as TSV: {e}")

    if len(df) == 0:
        raise ValidationError(f"{name}: file is empty (no data rows)")

    if required_columns:
        missing = [c for c in required_columns
                   if c not in df.columns]
        if missing:
            raise ValidationError(
                f"{name}: missing required columns: {missing}. "
                f"Found: {list(df.columns)}")

    # Count full rows
    with open(path) as fh:
        n_rows = sum(1 for _ in fh) - 1  # subtract header

    return {
        'path': str(path),
        'n_rows': n_rows,
        'columns': list(df.columns),
        'valid': True,
    }


def validate_bam_file(path, require_index=True,
                       require_cb_tag=True):
    """Validate a BAM file.

    :param path: Path to BAM.
    :param require_index: check for .bai index.
    :param require_cb_tag: check first reads for CB tag.
    :returns: dict with validation results.
    """
    logger = get_named_logger("Validate")
    path = Path(path)
    warnings = []

    if not path.exists():
        raise ValidationError(f"BAM file not found: {path}")

    has_index = False
    bai_path = Path(f"{path}.bai")
    bai_path2 = path.with_suffix('.bai')
    if bai_path.exists() or bai_path2.exists():
        has_index = True
    elif require_index:
        raise ValidationError(
            f"BAM index not found: {bai_path} "
            f"(run: samtools index {path})")

    has_cb = False
    if require_cb_tag:
        try:
            import pysam
            bam = pysam.AlignmentFile(str(path), 'rb')
            n_checked = 0
            for read in bam:
                try:
                    read.get_tag('CB')
                    has_cb = True
                    break
                except KeyError:
                    pass
                n_checked += 1
                if n_checked >= 100:
                    break
            bam.close()
            if not has_cb:
                warnings.append(
                    "No CB (cell barcode) tag found in first "
                    "100 reads. ASE module requires CB tags.")
        except ImportError:
            warnings.append(
                "pysam not installed — cannot verify CB tags")

    logger.info(
        f"  BAM: {path} (index: {has_index}, CB tag: {has_cb})")

    return {
        'path': str(path),
        'has_index': has_index,
        'has_cb_tag': has_cb,
        'valid': True,
        'warnings': warnings,
    }


def _load_barcodes_list(mex_dir):
    """Load barcodes from a MEX directory as a list (preserving duplicates)."""
    mex_dir = Path(mex_dir)
    for fname in ['barcodes.tsv.gz', 'barcodes.tsv']:
        p = mex_dir / fname
        if p.exists():
            if str(p).endswith('.gz'):
                with gzip.open(p, 'rt') as fh:
                    return [line.strip() for line in fh]
            else:
                with open(p) as fh:
                    return [line.strip() for line in fh]
    return []


def validate_barcode_overlap(gene_dir, transcript_dir):
    """Check barcode overlap between gene and transcript matrices.

    :returns: dict with overlap statistics.
    """
    logger = get_named_logger("Validate")

    gene_bc_list = _load_barcodes_list(gene_dir)
    tx_bc_list = _load_barcodes_list(transcript_dir)

    # Check for duplicate barcodes
    gene_bc = set(gene_bc_list)
    tx_bc = set(tx_bc_list)
    if len(gene_bc_list) != len(gene_bc):
        n_dup = len(gene_bc_list) - len(gene_bc)
        logger.warning(
            f"  Gene matrix has {n_dup} duplicate barcodes.")
    if len(tx_bc_list) != len(tx_bc):
        n_dup = len(tx_bc_list) - len(tx_bc)
        logger.warning(
            f"  Transcript matrix has {n_dup} duplicate barcodes.")
    overlap = gene_bc & tx_bc
    pct = len(overlap) / max(len(gene_bc), 1) * 100

    logger.info(
        f"  Barcode overlap: {len(overlap)}/{len(gene_bc)} "
        f"({pct:.1f}%)")

    if pct < 80:
        logger.warning(
            f"  Low barcode overlap ({pct:.1f}%) between gene and "
            f"transcript matrices. Check they come from the same run.")

    return {
        'n_gene_barcodes': len(gene_bc),
        'n_transcript_barcodes': len(tx_bc),
        'n_overlap': len(overlap),
        'pct_overlap': round(pct, 1),
    }


def validate_pipeline_inputs(args):
    """Validate all pipeline inputs.

    :param args: argparse namespace with pipeline arguments.
    :raises ValidationError: if critical inputs are invalid.
    """
    logger = get_named_logger("Validate")
    logger.info("Validating inputs...")

    all_warnings = []

    # Gene matrix (required)
    gene_dir = getattr(args, 'gene_matrix_dir', None)
    if gene_dir:
        result = validate_mex_directory(gene_dir, "Gene matrix")
        all_warnings.extend(result.get('warnings', []))

    # Transcript matrix (required for most modules)
    tx_dir = getattr(args, 'transcript_matrix_dir', None)
    if tx_dir:
        result = validate_mex_directory(tx_dir, "Transcript matrix")
        all_warnings.extend(result.get('warnings', []))

        # Barcode overlap check
        if gene_dir:
            validate_barcode_overlap(gene_dir, tx_dir)

    # Gene-transcript map
    map_file = getattr(args, 'gene_transcript_map', None)
    if map_file and Path(map_file).exists():
        validate_tsv_file(map_file, name="Gene-transcript map")

    # BAM file
    bam_file = getattr(args, 'tagged_bam', None)
    if bam_file and Path(bam_file).exists():
        result = validate_bam_file(bam_file)
        all_warnings.extend(result.get('warnings', []))

    # Annotated GTFs
    gtf_files = getattr(args, 'annotated_gtfs', None)
    if gtf_files:
        for gtf in gtf_files:
            if not Path(gtf).exists():
                raise ValidationError(
                    f"Annotated GTF not found: {gtf}")

    if all_warnings:
        for w in all_warnings:
            logger.warning(f"  WARNING: {w}")

    logger.info("Input validation complete.")
    return all_warnings


def main(args):
    """Run standalone input validation."""
    logger = get_named_logger("Validate")
    logger.info("IRIS input validation")

    try:
        validate_pipeline_inputs(args)
        logger.info("All inputs valid.")
    except ValidationError as e:
        logger.error(f"Validation FAILED: {e}")
        raise SystemExit(1)
