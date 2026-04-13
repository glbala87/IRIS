"""Allele-specific expression analysis using long-read phasing.

Detects allelic imbalance at heterozygous sites using single-cell
long-read data. Supports variant discovery from BAM pileups or
loading pre-called variants from VCF. Tests for allelic imbalance
per cell and per cluster using binomial and Fisher's exact tests.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from tqdm import tqdm

from ._logging import get_named_logger, wf_parser


def argparser():
    """Create argument parser."""
    parser = wf_parser("allele_specific_expression")

    parser.add_argument(
        "tagged_bam", type=Path,
        help="BAM file with cell barcode (CB) tags.")
    parser.add_argument(
        "--vcf", type=Path, default=None,
        help="VCF file with heterozygous variant calls. "
             "If not provided, variants are discovered from BAM pileups.")
    parser.add_argument(
        "--clusters", type=Path, required=True,
        help="TSV with columns barcode, cluster.")
    parser.add_argument(
        "--output_ase", type=Path, default="ase_results.tsv",
        help="Output TSV with per-variant allele-specific expression.")
    parser.add_argument(
        "--output_summary", type=Path, default="ase_summary.json",
        help="Output JSON with ASE summary statistics.")
    parser.add_argument(
        "--min_total_counts", type=int, default=10,
        help="Minimum total allelic counts per variant per cell/cluster.")
    parser.add_argument(
        "--min_cells", type=int, default=5,
        help="Minimum cells covering a variant site.")
    parser.add_argument(
        "--fdr_threshold", type=float, default=0.05,
        help="FDR threshold for significant allelic imbalance.")
    parser.add_argument(
        "--min_base_quality", type=int, default=20,
        help="Minimum base quality for allele counting.")
    parser.add_argument(
        "--het_threshold", type=float, default=0.20,
        help="Minimum allele frequency to call a heterozygous variant.")
    parser.add_argument(
        "--min_cov", type=int, default=10,
        help="Minimum coverage to evaluate a position for variant discovery.")
    parser.add_argument(
        "--n_jobs", type=int, default=1,
        help="Number of parallel worker processes for Fisher's exact tests. "
             "Use -1 to use all available CPUs.")

    return parser


def extract_variants_from_bam(bam_path, min_bq=20, min_cov=10, het_threshold=0.20):
    """Discover heterozygous positions from BAM pileups.

    Scans pileup columns and identifies positions where two alleles
    each comprise at least het_threshold fraction of reads. This heuristic captures
    likely heterozygous SNVs without requiring a VCF.

    :param bam_path: Path to indexed BAM file.
    :param min_bq: minimum base quality to count an allele.
    :param min_cov: minimum coverage to evaluate a position.
    :param het_threshold: minimum allele frequency to call heterozygous.
    :returns: DataFrame with columns chrom, pos, ref, alt.
    """
    try:
        import pysam
    except ImportError:
        raise ImportError(
            "pysam is required for ASE analysis. "
            "Install with: pip install iris-sc[ase]")

    logger = get_named_logger("Variants")
    logger.info(f"Discovering heterozygous variants from {bam_path}.")

    variants = []
    try:
        bam = pysam.AlignmentFile(str(bam_path), 'rb')
    except (OSError, ValueError) as e:
        logger.error(f"Cannot open BAM file {bam_path}: {e}")
        return pd.DataFrame(columns=['chrom', 'pos', 'ref', 'alt'])

    try:
        for pileup_col in bam.pileup(
                min_base_quality=min_bq,
                truncate=True,
                stepper='nofilter'):

            chrom = pileup_col.reference_name
            pos = pileup_col.reference_pos

            allele_counts = {}
            for read in pileup_col.pileups:
                if read.is_del or read.is_refskip:
                    continue
                base = read.alignment.query_sequence[read.query_position]
                base = base.upper()
                if base not in 'ACGT':
                    continue
                allele_counts[base] = allele_counts.get(base, 0) + 1

            total = sum(allele_counts.values())
            if total < min_cov:
                continue

            # Identify two most common alleles
            sorted_alleles = sorted(
                allele_counts.items(), key=lambda x: -x[1])
            if len(sorted_alleles) < 2:
                continue

            top_base, top_count = sorted_alleles[0]
            sec_base, sec_count = sorted_alleles[1]

            # Both alleles must be at least het_threshold frequency
            if (top_count / total >= het_threshold
                    and sec_count / total >= het_threshold):
                variants.append({
                    'chrom': chrom,
                    'pos': pos,
                    'ref': top_base,
                    'alt': sec_base,
                })
    finally:
        bam.close()

    variant_df = pd.DataFrame(variants)
    logger.info(f"Discovered {len(variant_df)} heterozygous positions.")
    return variant_df


def load_variants_from_vcf(vcf_path):
    """Parse VCF file for heterozygous variant sites.

    Extracts positions where the genotype (GT) is heterozygous (0/1 or
    0|1). Only single-nucleotide variants are retained.

    :param vcf_path: Path to VCF file.
    :returns: DataFrame with columns chrom, pos, ref, alt.
    """
    logger = get_named_logger("VCF")
    logger.info(f"Loading variants from {vcf_path}.")

    variants = []
    path = Path(vcf_path)

    if path.suffix in ('.gz', '.bgz'):
        import gzip
        opener = gzip.open(path, 'rt')
    else:
        opener = open(path, 'r')

    with opener as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 10:
                continue

            chrom = fields[0]
            pos = int(fields[1]) - 1  # Convert to 0-based
            ref = fields[3]
            alt_field = fields[4]

            # Only keep SNVs; handle multi-allelic sites by selecting the
            # single SNV alt allele when exactly one alt allele is an SNV.
            alt_alleles = alt_field.split(',')
            snv_alts = [
                a for a in alt_alleles
                if len(ref) == 1 and len(a) == 1 and a not in ('.', '*')
            ]
            if len(snv_alts) != 1:
                continue
            alt = snv_alts[0]

            # Check genotype for heterozygosity
            format_fields = fields[8].split(':')
            sample_fields = fields[9].split(':')
            try:
                gt_idx = format_fields.index('GT')
                gt = sample_fields[gt_idx]
            except (ValueError, IndexError):
                # FORMAT does not contain GT or sample field is truncated
                continue

            # Skip haploid genotypes (e.g. GT='0' or GT='1')
            if '/' not in gt and '|' not in gt:
                continue

            if gt in ('0/1', '1/0', '0|1', '1|0'):
                variants.append({
                    'chrom': chrom,
                    'pos': pos,
                    'ref': ref,
                    'alt': alt,
                })

    variant_df = pd.DataFrame(variants)
    logger.info(f"Loaded {len(variant_df)} heterozygous SNVs from VCF.")
    return variant_df


def count_alleles_per_cell(bam_path, variants, min_bq=20):
    """Count reference and alternative alleles per cell per variant.

    For each variant site, iterates over covering reads. If a read
    carries a CB (cell barcode) tag, its allele is counted for that
    cell.

    :param bam_path: Path to indexed BAM file.
    :param variants: DataFrame with chrom, pos, ref, alt.
    :param min_bq: minimum base quality for counting.
    :returns: DataFrame with chrom, pos, barcode, ref_count, alt_count.
    """
    try:
        import pysam
    except ImportError:
        raise ImportError(
            "pysam is required for ASE analysis. "
            "Install with: pip install iris-sc[ase]")

    logger = get_named_logger("AlleleCt")
    logger.info(
        f"Counting alleles per cell at {len(variants)} variant sites.")

    try:
        bam = pysam.AlignmentFile(str(bam_path), 'rb')
    except (OSError, ValueError) as e:
        logger.error(f"Cannot open BAM file {bam_path}: {e}")
        return pd.DataFrame(
            columns=['chrom', 'pos', 'barcode', 'ref_count', 'alt_count'])

    records = []

    try:
        for _, var in tqdm(variants.iterrows(), total=len(variants), desc="Counting alleles", disable=None):
            chrom = var['chrom']
            pos = int(var['pos'])
            ref_base = var['ref'].upper()
            alt_base = var['alt'].upper()

            cell_alleles = {}  # barcode -> [ref_count, alt_count]

            for read in bam.fetch(chrom, pos, pos + 1):
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue

                # Get cell barcode
                try:
                    barcode = read.get_tag('CB')
                except KeyError:
                    continue

                # Get base at variant position using aligned pairs
                # Use with_seq=False to avoid requiring MD tags
                pairs = read.get_aligned_pairs(with_seq=False)
                for query_pos, ref_pos in pairs:
                    if ref_pos == pos and query_pos is not None:
                        if read.query_qualities is not None:
                            base_qual = read.query_qualities[query_pos]
                            if base_qual < min_bq:
                                break
                        query_base = read.query_sequence[query_pos].upper()
                        if barcode not in cell_alleles:
                            cell_alleles[barcode] = [0, 0]
                        if query_base == ref_base:
                            cell_alleles[barcode][0] += 1
                        elif query_base == alt_base:
                            cell_alleles[barcode][1] += 1
                        break

            for barcode, (rc, ac) in cell_alleles.items():
                if rc + ac > 0:
                    records.append({
                        'chrom': chrom,
                        'pos': pos,
                        'barcode': barcode,
                        'ref_count': rc,
                        'alt_count': ac,
                    })
    finally:
        bam.close()

    allele_df = pd.DataFrame(records)
    logger.info(
        f"Counted alleles in {len(allele_df)} cell-variant pairs "
        f"across {allele_df['barcode'].nunique() if len(allele_df) > 0 else 0}"
        f" cells.")
    return allele_df


def compute_allelic_imbalance(ref_count, alt_count):
    """Test for allelic imbalance at a single site using binomial test.

    Tests the null hypothesis that both alleles are equally represented
    (p = 0.5). Uses scipy.stats.binomtest.

    :param ref_count: number of reference allele observations.
    :param alt_count: number of alternative allele observations.
    :returns: tuple (pvalue, allelic_ratio) where allelic_ratio = alt/(ref+alt).
    """
    total = ref_count + alt_count
    if total == 0:
        return 1.0, np.nan

    result = scipy.stats.binomtest(alt_count, total, 0.5)
    pvalue = result.pvalue
    allelic_ratio = alt_count / total
    return pvalue, allelic_ratio


def _fisher_test_variant(chrom, pos, cluster_data, unique_clusters):
    """Run Fisher's exact test for all cluster pairs at one variant site.

    Intended to be called in a worker process via ProcessPoolExecutor.

    :param chrom: chromosome string.
    :param pos: integer position.
    :param cluster_data: dict mapping cluster label ->
        (ref_count, alt_count) tuple.
    :param unique_clusters: sorted list of cluster labels to test.
    :returns: list of diff-ASE result dicts for this variant.
    """
    rows = []
    for i, c1 in enumerate(unique_clusters):
        for c2 in unique_clusters[i + 1:]:
            if c1 not in cluster_data or c2 not in cluster_data:
                continue
            r1, a1 = cluster_data[c1]
            r2, a2 = cluster_data[c2]
            if r1 + a1 == 0 or r2 + a2 == 0:
                continue
            _, fisher_p = scipy.stats.fisher_exact([[r1, a1], [r2, a2]])
            rows.append({
                'chrom': chrom,
                'pos': int(pos),
                'cluster_1': c1,
                'cluster_2': c2,
                'ref_1': r1, 'alt_1': a1,
                'ref_2': r2, 'alt_2': a2,
                'ratio_1': round(a1 / (r1 + a1), 4),
                'ratio_2': round(a2 / (r2 + a2), 4),
                'fisher_pvalue': fisher_p,
            })
    return rows


def aggregate_by_cluster(allele_counts, cluster_labels, n_jobs=1):
    """Aggregate allele counts per cluster and test for imbalance.

    For each variant in each cluster, sums allele counts across cells
    and tests for allelic imbalance. Additionally performs Fisher's exact
    test for differential ASE between all pairs of clusters.

    :param allele_counts: DataFrame from count_alleles_per_cell.
    :param cluster_labels: Series mapping barcode -> cluster.
    :param n_jobs: number of parallel worker processes for Fisher's exact
        tests. 1 = sequential. Pass -1 to use all available CPUs.
    :returns: tuple (cluster_ase_df, differential_ase_df).
    """
    logger = get_named_logger("ClusterASE")

    if len(allele_counts) == 0:
        empty = pd.DataFrame()
        return empty, empty

    # Map barcodes to clusters
    allele_counts = allele_counts.copy()
    allele_counts['cluster'] = allele_counts['barcode'].map(cluster_labels)
    allele_counts = allele_counts.dropna(subset=['cluster'])

    if len(allele_counts) == 0:
        logger.warning("No allele counts mapped to clusters.")
        empty = pd.DataFrame()
        return empty, empty

    # Aggregate per variant per cluster
    grouped = allele_counts.groupby(['chrom', 'pos', 'cluster']).agg(
        ref_count=('ref_count', 'sum'),
        alt_count=('alt_count', 'sum'),
        n_cells=('barcode', 'nunique'),
    ).reset_index()

    # Test imbalance per cluster
    cluster_rows = []
    for _, row in grouped.iterrows():
        rc = int(row['ref_count'])
        ac = int(row['alt_count'])
        pval, ratio = compute_allelic_imbalance(rc, ac)
        cluster_rows.append({
            'chrom': row['chrom'],
            'pos': int(row['pos']),
            'cluster': row['cluster'],
            'ref_count': rc,
            'alt_count': ac,
            'total_count': rc + ac,
            'n_cells': int(row['n_cells']),
            'allelic_ratio': round(ratio, 4) if not np.isnan(ratio) else np.nan,
            'pvalue': pval,
        })

    cluster_ase_df = pd.DataFrame(cluster_rows)

    # BH correction
    if len(cluster_ase_df) > 0:
        cluster_ase_df['pvalue_adj'] = _bh_correct(
            cluster_ase_df['pvalue'].values)

    # Differential ASE between cluster pairs (Fisher's exact)
    diff_rows = []
    unique_clusters = sorted(cluster_ase_df['cluster'].unique())
    variant_groups = cluster_ase_df.groupby(['chrom', 'pos'])

    # Build per-variant argument tuples for (potential) parallel dispatch
    variant_args = []
    for (chrom, pos), vgrp in variant_groups:
        cluster_data = {
            row['cluster']: (int(row['ref_count']), int(row['alt_count']))
            for _, row in vgrp.iterrows()
        }
        variant_args.append((chrom, pos, cluster_data, unique_clusters))

    if n_jobs == -1:
        import os
        n_jobs = os.cpu_count() or 1

    if n_jobs == 1:
        for chrom, pos, cluster_data, uc in variant_args:
            diff_rows.extend(
                _fisher_test_variant(chrom, pos, cluster_data, uc))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    _fisher_test_variant, chrom, pos, cluster_data, uc)
                for chrom, pos, cluster_data, uc in variant_args
            ]
            for f in futures:
                diff_rows.extend(f.result())

    diff_ase_df = pd.DataFrame(diff_rows)
    if len(diff_ase_df) > 0:
        diff_ase_df['fisher_pvalue_adj'] = _bh_correct(
            diff_ase_df['fisher_pvalue'].values)

    n_sig_cluster = 0
    n_sig_diff = 0
    if len(cluster_ase_df) > 0:
        n_sig_cluster = int(
            (cluster_ase_df['pvalue_adj'] <= 0.05).sum())
    if len(diff_ase_df) > 0:
        n_sig_diff = int(
            (diff_ase_df['fisher_pvalue_adj'] <= 0.05).sum())

    logger.info(
        f"Cluster ASE: {n_sig_cluster} significant imbalanced "
        f"variant-cluster pairs; {n_sig_diff} differential ASE pairs.")
    return cluster_ase_df, diff_ase_df


from ._stats import bh_correct as _bh_correct


def main(args):
    """Run allele-specific expression analysis pipeline."""
    logger = get_named_logger("ASE")
    logger.info("Starting allele-specific expression analysis.")

    # Discover or load variants
    if args.vcf is not None:
        variants = load_variants_from_vcf(args.vcf)
    else:
        variants = extract_variants_from_bam(
            args.tagged_bam,
            min_bq=args.min_base_quality,
            min_cov=args.min_cov,
            het_threshold=args.het_threshold)

    if len(variants) == 0:
        logger.warning("No heterozygous variants found. Exiting.")
        # Write empty outputs
        pd.DataFrame(columns=[
            'chrom', 'pos', 'cluster', 'ref_count', 'alt_count',
            'total_count', 'n_cells', 'allelic_ratio', 'pvalue', 'pvalue_adj'
        ]).to_csv(args.output_ase, sep='\t', index=False)
        with open(args.output_summary, 'w') as fh:
            json.dump({'n_variants': 0, 'n_significant': 0}, fh, indent=2)
        return

    logger.info(f"Proceeding with {len(variants)} variant sites.")

    # Count alleles per cell
    allele_counts = count_alleles_per_cell(
        args.tagged_bam, variants, min_bq=args.min_base_quality)

    if len(allele_counts) == 0:
        logger.warning("No allele counts obtained. Exiting.")
        pd.DataFrame(columns=[
            'chrom', 'pos', 'cluster', 'ref_count', 'alt_count',
            'total_count', 'n_cells', 'allelic_ratio', 'pvalue', 'pvalue_adj'
        ]).to_csv(args.output_ase, sep='\t', index=False)
        with open(args.output_summary, 'w') as fh:
            json.dump({'n_variants': len(variants), 'n_significant': 0},
                      fh, indent=2)
        return

    # Filter by minimum total counts per variant-cell
    allele_counts['total'] = (
        allele_counts['ref_count'] + allele_counts['alt_count'])
    allele_counts = allele_counts[
        allele_counts['total'] >= args.min_total_counts].copy()
    allele_counts = allele_counts.drop(columns=['total'])

    # Filter variants by minimum cells
    variant_cell_counts = allele_counts.groupby(
        ['chrom', 'pos'])['barcode'].nunique().reset_index()
    variant_cell_counts.columns = ['chrom', 'pos', 'n_cells']
    passing_variants = variant_cell_counts[
        variant_cell_counts['n_cells'] >= args.min_cells]
    allele_counts = allele_counts.merge(
        passing_variants[['chrom', 'pos']], on=['chrom', 'pos'])

    logger.info(
        f"After filtering: {allele_counts['barcode'].nunique()} cells, "
        f"{len(passing_variants)} variants.")

    # Load cluster assignments
    clusters_df = pd.read_csv(args.clusters, sep='\t')
    if len(clusters_df.columns) < 2:
        raise ValueError(
            f"Clusters file '{args.clusters}' must have at least 2 columns "
            f"(barcode and cluster), but only {len(clusters_df.columns)} "
            f"column(s) found: {list(clusters_df.columns)}. "
            "Expected a tab-separated file with a barcode column followed "
            "by a cluster column.")
    if len(clusters_df.columns) > 2:
        logger.warning(
            f"Clusters file has {len(clusters_df.columns)} columns; "
            f"using '{clusters_df.columns[0]}' as barcode and "
            f"'{clusters_df.columns[1]}' as cluster.")
    barcode_col = clusters_df.columns[0]
    cluster_col = clusters_df.columns[1]
    cluster_labels = pd.Series(
        clusters_df[cluster_col].astype(str).values,
        index=clusters_df[barcode_col].values)

    # Aggregate by cluster and test
    cluster_ase_df, diff_ase_df = aggregate_by_cluster(
        allele_counts, cluster_labels,
        n_jobs=getattr(args, 'n_jobs', 1))

    # Write results
    if len(cluster_ase_df) > 0:
        cluster_ase_df.to_csv(args.output_ase, sep='\t', index=False)
    else:
        pd.DataFrame(columns=[
            'chrom', 'pos', 'cluster', 'ref_count', 'alt_count',
            'total_count', 'n_cells', 'allelic_ratio', 'pvalue', 'pvalue_adj'
        ]).to_csv(args.output_ase, sep='\t', index=False)

    # Write differential ASE alongside
    diff_output = str(args.output_ase).replace(
        '.tsv', '_differential.tsv')
    if len(diff_ase_df) > 0:
        diff_ase_df.to_csv(diff_output, sep='\t', index=False)

    # Summary
    n_variants_tested = 0
    n_sig_imbalance = 0
    n_sig_differential = 0

    if len(cluster_ase_df) > 0:
        n_variants_tested = int(
            cluster_ase_df.groupby(['chrom', 'pos']).ngroups)
        n_sig_imbalance = int(
            (cluster_ase_df['pvalue_adj'] <= args.fdr_threshold).sum())
    if len(diff_ase_df) > 0:
        n_sig_differential = int(
            (diff_ase_df['fisher_pvalue_adj'] <= args.fdr_threshold).sum())

    summary = {
        'n_variants_input': len(variants),
        'n_variants_passing_filters': int(len(passing_variants)),
        'n_variants_tested': n_variants_tested,
        'n_cells_with_allele_counts': int(
            allele_counts['barcode'].nunique()),
        'n_significant_imbalance': n_sig_imbalance,
        'n_significant_differential': n_sig_differential,
        'fdr_threshold': args.fdr_threshold,
        'min_total_counts': args.min_total_counts,
        'min_cells': args.min_cells,
        'variant_source': 'vcf' if args.vcf else 'bam_pileup',
    }
    with open(args.output_summary, 'w') as fh:
        json.dump(summary, fh, indent=2)

    logger.info(
        f"ASE analysis complete: {n_variants_tested} variants tested, "
        f"{n_sig_imbalance} with significant imbalance, "
        f"{n_sig_differential} with differential ASE between clusters.")
