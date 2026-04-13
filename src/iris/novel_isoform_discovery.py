"""Novel isoform discovery per cell cluster (isosceles Module 6).

Identifies novel transcript isoforms from StringTie assemblies that are
enriched in specific cell clusters. Uses gffcompare class codes to
classify transcripts as known or novel, cross-references with
per-cell transcript counts, and tests for cluster-specific enrichment.

Novel class codes of interest:
  j - novel isoform sharing at least one splice junction with reference
  o - generic exonic overlap with reference
  x - exonic overlap on opposite strand (antisense)
  u - intergenic (completely novel, no reference overlap)
  e - single exon partially covering an intron
  n - novel intronic (overlaps intron of reference on same strand)

Reference: Pertea & Pertea, GFFCompare (2020), gffcompare documentation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats
from tqdm import tqdm

from ._logging import get_named_logger, wf_parser

# gffcompare class codes indicating novel isoforms
NOVEL_CLASS_CODES = {'j', 'o', 'x', 'u', 'e', 'n', 'i', 's'}

# Known/reference class codes
KNOWN_CLASS_CODES = {'=', 'c'}


def argparser():
    """Create argument parser."""
    parser = wf_parser("novel_isoform_discovery")

    parser.add_argument(
        "transcript_matrix_dir", type=Path,
        help="Path to processed transcript MEX matrix directory.")
    parser.add_argument(
        "--annotated_gtfs", type=Path, nargs='+', required=True,
        help="Gffcompare annotated GTF files (one per chromosome).")
    parser.add_argument(
        "--clusters", type=Path, required=True,
        help="TSV with columns barcode, cluster.")
    parser.add_argument(
        "--gene_transcript_map", type=Path, required=True,
        help="TSV mapping transcript_id to gene_id.")
    parser.add_argument(
        "--output_novel_catalog", type=Path,
        default="novel_isoform_catalog.tsv",
        help="Output TSV cataloging all novel isoforms with class codes.")
    parser.add_argument(
        "--output_cluster_enrichment", type=Path,
        default="novel_isoform_enrichment.tsv",
        help="Output TSV with cluster-specific novel isoform enrichment.")
    parser.add_argument(
        "--output_summary", type=Path,
        default="novel_isoform_summary.json",
        help="Output JSON with discovery summary statistics.")
    parser.add_argument(
        "--min_cells", type=int, default=3,
        help="Minimum cells expressing a novel isoform to include.")
    parser.add_argument(
        "--min_counts", type=int, default=5,
        help="Minimum total UMI counts for a novel isoform.")
    parser.add_argument(
        "--enrichment_fdr", type=float, default=0.05,
        help="FDR threshold for cluster enrichment.")
    parser.add_argument(
        "--cluster_column", type=str, default="cluster",
        help="Column name for cluster labels.")

    return parser


def parse_gffcompare_gtf(gtf_path):
    """Parse a gffcompare annotated GTF to extract transcript metadata.

    Extracts transcript_id, gene_id, class_code, and reference IDs
    from the attributes field of transcript-level entries.

    :param gtf_path: Path to a gffcompare annotated GTF file.
    :returns: DataFrame with columns transcript_id, gene_id,
        class_code, ref_gene_id, ref_transcript_id, chrom.
    """
    import re

    records = []
    path = Path(gtf_path)

    # Handle gzipped files
    if path.suffix == '.gz':
        import gzip
        opener = gzip.open(path, 'rt')
    else:
        opener = open(path, 'r')

    with opener as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            if fields[2] != 'transcript':
                continue

            chrom = fields[0]
            attrs = fields[8]

            # Extract key attributes
            tx_match = re.search(r'transcript_id "([^"]+)"', attrs)
            gene_match = re.search(r'gene_id "([^"]+)"', attrs)
            class_match = re.search(r'class_code "([^"]+)"', attrs)
            ref_gene_match = re.search(r'ref_gene_id "([^"]+)"', attrs)
            ref_tx_match = re.search(
                r'cmp_ref "([^"]+)"', attrs)

            if tx_match is None:
                continue

            records.append({
                'transcript_id': tx_match.group(1),
                'gene_id': gene_match.group(1) if gene_match else '',
                'class_code': class_match.group(1) if class_match else '',
                'ref_gene_id': ref_gene_match.group(1)
                    if ref_gene_match else '',
                'ref_transcript_id': ref_tx_match.group(1)
                    if ref_tx_match else '',
                'chrom': chrom,
            })

    return pd.DataFrame(records)


def parse_multiple_gtfs(gtf_paths):
    """Parse multiple gffcompare annotated GTFs and combine.

    :param gtf_paths: list of Path objects to GTF files.
    :returns: combined DataFrame with all transcripts.
    """
    logger = get_named_logger("GTFParse")
    all_dfs = []

    for gtf_path in gtf_paths:
        df = parse_gffcompare_gtf(gtf_path)
        if len(df) > 0:
            all_dfs.append(df)

    if not all_dfs:
        logger.warning("No transcripts found in any GTF file.")
        return pd.DataFrame(columns=[
            'transcript_id', 'gene_id', 'class_code',
            'ref_gene_id', 'ref_transcript_id', 'chrom'])

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset='transcript_id')
    logger.info(
        f"Parsed {len(combined)} transcripts from "
        f"{len(gtf_paths)} GTF files.")
    return combined


def classify_transcripts(annotation_df):
    """Classify transcripts as novel or known based on class code.

    :param annotation_df: DataFrame from parse_gffcompare_gtf.
    :returns: annotation_df with added 'novelty' column.
    """
    def _classify(code):
        if code in KNOWN_CLASS_CODES:
            return 'known'
        elif code in NOVEL_CLASS_CODES:
            return 'novel'
        else:
            return 'ambiguous'

    annotation_df = annotation_df.copy()
    annotation_df['novelty'] = annotation_df['class_code'].apply(_classify)
    return annotation_df


def build_novel_catalog(annotation_df, adata, min_cells=3, min_counts=5):
    """Build a catalog of novel isoforms with expression statistics.

    Cross-references novel transcripts from gffcompare with the
    transcript expression matrix to get per-isoform statistics.

    :param annotation_df: classified transcript annotations.
    :param adata: AnnData with transcript-level counts.
    :param min_cells: minimum cells expressing the isoform.
    :param min_counts: minimum total UMI counts.
    :returns: DataFrame catalog of novel isoforms.
    """
    logger = get_named_logger("Catalog")

    novel_df = annotation_df[
        annotation_df['novelty'] == 'novel'].copy()
    logger.info(f"Found {len(novel_df)} novel transcripts in annotations.")

    if len(novel_df) == 0:
        return pd.DataFrame()

    # Find which novel transcripts are in the expression matrix
    matrix_tx = set(adata.var_names)
    novel_in_matrix = novel_df[
        novel_df['transcript_id'].isin(matrix_tx)].copy()
    logger.info(
        f"{len(novel_in_matrix)} novel transcripts found in "
        f"expression matrix (out of {len(novel_df)}).")

    if len(novel_in_matrix) == 0:
        return pd.DataFrame()

    # Build O(1) index for transcript lookups
    var_name_to_idx = {
        name: idx for idx, name in enumerate(adata.var_names)}

    # Batch-extract all novel transcript indices
    novel_tx_ids = novel_in_matrix['transcript_id'].values
    novel_indices = [var_name_to_idx[tx] for tx in novel_tx_ids
                     if tx in var_name_to_idx]
    if not novel_indices:
        return pd.DataFrame()

    # Extract all novel columns at once (single sparse slice)
    X = adata.X
    if scipy.sparse.issparse(X):
        novel_block = safe_toarray(
            X[:, novel_indices], context="novel isoform expression block")
    else:
        novel_block = np.asarray(X[:, novel_indices])

    catalog_rows = []
    col_idx = 0
    for _, row in novel_in_matrix.iterrows():
        tx_id = row['transcript_id']
        if tx_id not in var_name_to_idx:
            continue

        col = novel_block[:, col_idx]
        col_idx += 1

        n_cells = int((col > 0).sum())
        total_counts = int(col.sum())
        mean_expr = float(col[col > 0].mean()) if n_cells > 0 else 0.0

        if n_cells < min_cells or total_counts < min_counts:
            continue

        catalog_rows.append({
            'transcript_id': tx_id,
            'gene_id': row['gene_id'],
            'class_code': row['class_code'],
            'ref_gene_id': row['ref_gene_id'],
            'ref_transcript_id': row['ref_transcript_id'],
            'chrom': row['chrom'],
            'n_cells': n_cells,
            'total_counts': total_counts,
            'mean_expr_per_cell': round(mean_expr, 4),
        })

    catalog = pd.DataFrame(catalog_rows)
    if len(catalog) > 0:
        catalog = catalog.sort_values(
            'total_counts', ascending=False).reset_index(drop=True)

    logger.info(
        f"Novel isoform catalog: {len(catalog)} isoforms passing "
        f"filters (min_cells={min_cells}, min_counts={min_counts}).")
    return catalog


def test_cluster_enrichment(
        adata, novel_catalog, cluster_labels, fdr_threshold=0.05):
    """Test whether novel isoforms are enriched in specific clusters.

    For each novel isoform, tests whether cells expressing it are
    over-represented in any cluster using Fisher's exact test.

    :param adata: AnnData with transcript-level counts.
    :param novel_catalog: DataFrame from build_novel_catalog.
    :param cluster_labels: Series mapping barcode to cluster.
    :param fdr_threshold: FDR threshold for significance.
    :returns: DataFrame with enrichment results.
    """
    logger = get_named_logger("Enrichment")

    if len(novel_catalog) == 0:
        return pd.DataFrame()

    # Align barcodes
    common = sorted(
        set(adata.obs_names) & set(cluster_labels.index))
    if len(common) == 0:
        logger.warning("No overlapping barcodes.")
        return pd.DataFrame()

    adata_sub = adata[common].copy()
    clusters = cluster_labels.loc[common]
    unique_clusters = sorted(clusters.unique())
    n_total = len(common)

    X = adata_sub.X
    sub_var_name_to_idx = {
        name: idx for idx, name in enumerate(adata_sub.var_names)}
    enrichment_rows = []

    for _, row in tqdm(novel_catalog.iterrows(), total=len(novel_catalog), desc="Testing enrichment", disable=None):
        tx_id = row['transcript_id']
        if tx_id not in sub_var_name_to_idx:
            continue

        tx_idx = sub_var_name_to_idx[tx_id]

        if scipy.sparse.issparse(X):
            expr = np.asarray(X[:, tx_idx].todense()).ravel()
        else:
            expr = np.asarray(X[:, tx_idx]).ravel()

        expressing_mask = expr > 0
        n_expressing = int(expressing_mask.sum())

        if n_expressing == 0:
            continue

        # Test each cluster for enrichment
        for cluster in unique_clusters:
            cluster_mask = np.array(clusters == cluster)
            n_cluster = int(cluster_mask.sum())

            # 2x2 contingency table:
            # [expressing & in_cluster, expressing & not_in_cluster]
            # [not_expressing & in_cluster, not_expressing & not_in_cluster]
            a = int((expressing_mask & cluster_mask).sum())
            b = int((expressing_mask & ~cluster_mask).sum())
            c = int((~expressing_mask & cluster_mask).sum())
            d = int((~expressing_mask & ~cluster_mask).sum())

            # Fisher's exact test (one-sided: over-representation)
            _, pval = scipy.stats.fisher_exact(
                [[a, b], [c, d]], alternative='greater')

            # Fold enrichment
            expected = n_expressing * n_cluster / n_total
            fold_enrichment = a / expected if expected > 0 else 0.0

            # Proportion of expressing cells in this cluster
            pct_in_cluster = a / n_cluster if n_cluster > 0 else 0.0
            pct_in_rest = b / (n_total - n_cluster) \
                if (n_total - n_cluster) > 0 else 0.0

            enrichment_rows.append({
                'transcript_id': tx_id,
                'gene_id': row['gene_id'],
                'class_code': row['class_code'],
                'ref_gene_id': row['ref_gene_id'],
                'cluster': cluster,
                'n_expressing_in_cluster': a,
                'n_expressing_total': n_expressing,
                'n_cells_in_cluster': n_cluster,
                'pct_in_cluster': round(pct_in_cluster, 4),
                'pct_in_rest': round(pct_in_rest, 4),
                'fold_enrichment': round(fold_enrichment, 4),
                'pvalue': pval,
            })

    if not enrichment_rows:
        return pd.DataFrame()

    enrichment_df = pd.DataFrame(enrichment_rows)

    # BH correction
    pvals = enrichment_df['pvalue'].values
    enrichment_df['pvalue_adj'] = _bh_correct(pvals)

    # Sort by adjusted p-value
    enrichment_df = enrichment_df.sort_values(
        'pvalue_adj').reset_index(drop=True)

    n_sig = int((enrichment_df['pvalue_adj'] <= fdr_threshold).sum())
    logger.info(
        f"Cluster enrichment: {n_sig} significant associations "
        f"(FDR < {fdr_threshold}) from {len(enrichment_df)} tests.")

    return enrichment_df


from ._stats import bh_correct as _bh_correct, safe_toarray


def compute_specificity_score(enrichment_df, fdr_threshold=0.05):
    """Compute a cluster specificity score for each novel isoform.

    Uses a geometric combination of fold enrichment and cluster
    specificity: score = fold_enrichment ^ (1 - normalized_entropy).
    This preserves fold-change magnitude for moderately distributed
    isoforms while still penalizing broadly expressed ones.

    :param enrichment_df: DataFrame from test_cluster_enrichment.
    :param fdr_threshold: FDR threshold for significant enrichment.
    :returns: DataFrame with per-isoform specificity scores.
    """
    if len(enrichment_df) == 0:
        return pd.DataFrame()

    specificity_rows = []

    for tx_id, grp in enrichment_df.groupby('transcript_id'):
        sig = grp[grp['pvalue_adj'] <= fdr_threshold]
        max_fold = float(grp['fold_enrichment'].max())
        best_cluster = grp.loc[grp['fold_enrichment'].idxmax(), 'cluster']

        # Distribution of expressing cells across clusters
        counts = grp['n_expressing_in_cluster'].values.astype(float)
        total = counts.sum()
        n_clusters_tested = len(grp)
        if total > 0:
            props = counts / total
            props = props[props > 0]
            n_expressing_clusters = len(props)
            entropy = -np.sum(props * np.log2(props))
            max_entropy = np.log2(n_clusters_tested) \
                if n_clusters_tested > 1 else 1.0
            normalized_entropy = entropy / max_entropy \
                if max_entropy > 0 else 0.0
        else:
            normalized_entropy = 1.0
            n_expressing_clusters = 0

        # Combined specificity score: fold-enrichment weighted by
        # cluster selectivity. Uses log-fold to handle the range
        # gracefully: score = log2(fold+1) * (1 - entropy)
        # This gives 0 for uniformly distributed (entropy=1) and
        # scales with fold enrichment for specific isoforms.
        log_fold = np.log2(max_fold + 1.0)
        specificity = log_fold * (1.0 - normalized_entropy)

        meta = grp.iloc[0]
        specificity_rows.append({
            'transcript_id': tx_id,
            'gene_id': meta['gene_id'],
            'class_code': meta['class_code'],
            'ref_gene_id': meta['ref_gene_id'],
            'best_cluster': best_cluster,
            'max_fold_enrichment': round(max_fold, 4),
            'n_significant_clusters': len(sig),
            'specificity_score': round(specificity, 4),
            'n_expressing_total': int(meta['n_expressing_total']),
        })

    spec_df = pd.DataFrame(specificity_rows)
    spec_df = spec_df.sort_values(
        'specificity_score', ascending=False).reset_index(drop=True)
    return spec_df


def main(args):
    """Run novel isoform discovery pipeline."""
    import scanpy as sc

    logger = get_named_logger("NovelIso")
    logger.info("Starting novel isoform discovery.")

    # Load transcript expression matrix
    logger.info(
        f"Loading transcript matrix from {args.transcript_matrix_dir}.")
    adata = sc.read_10x_mtx(
        str(args.transcript_matrix_dir), var_names='gene_symbols')
    adata.var_names_make_unique()
    logger.info(
        f"Loaded: {adata.shape[0]} cells x {adata.shape[1]} transcripts.")

    # Parse gffcompare annotated GTFs
    logger.info(
        f"Parsing {len(args.annotated_gtfs)} annotated GTF files.")
    annotation_df = parse_multiple_gtfs(args.annotated_gtfs)
    annotation_df = classify_transcripts(annotation_df)

    n_novel = int((annotation_df['novelty'] == 'novel').sum())
    n_known = int((annotation_df['novelty'] == 'known').sum())
    logger.info(
        f"Classified transcripts: {n_known} known, {n_novel} novel.")

    # Build novel isoform catalog
    catalog = build_novel_catalog(
        annotation_df, adata,
        min_cells=args.min_cells,
        min_counts=args.min_counts)
    catalog.to_csv(args.output_novel_catalog, sep='\t', index=False)
    logger.info(
        f"Novel catalog: {len(catalog)} isoforms written to "
        f"{args.output_novel_catalog}.")

    # Load cluster assignments
    clusters_df = pd.read_csv(args.clusters, sep='\t')
    cluster_col = args.cluster_column
    barcode_col = clusters_df.columns[0]
    cluster_labels = pd.Series(
        clusters_df[cluster_col].astype(str).values,
        index=clusters_df[barcode_col].values)

    # Test cluster enrichment
    enrichment_df = test_cluster_enrichment(
        adata, catalog, cluster_labels,
        fdr_threshold=args.enrichment_fdr)

    # Compute specificity scores
    if len(enrichment_df) > 0:
        spec_df = compute_specificity_score(
            enrichment_df, fdr_threshold=args.enrichment_fdr)
        # Merge specificity into enrichment for output
        enrichment_out = enrichment_df.merge(
            spec_df[['transcript_id', 'specificity_score',
                      'best_cluster']],
            on='transcript_id', how='left',
            suffixes=('', '_spec'))
    else:
        enrichment_out = enrichment_df
        spec_df = pd.DataFrame()

    enrichment_out.to_csv(
        args.output_cluster_enrichment, sep='\t', index=False)
    logger.info(
        f"Enrichment results: {len(enrichment_out)} associations "
        f"written to {args.output_cluster_enrichment}.")

    # Summary statistics
    n_sig = 0
    n_cluster_specific = 0
    top_novel = []

    if len(enrichment_df) > 0:
        sig_mask = enrichment_df['pvalue_adj'] <= args.enrichment_fdr
        n_sig = int(sig_mask.sum())

    if len(spec_df) > 0:
        # Cluster-specific = significant enrichment in exactly 1 cluster
        n_cluster_specific = int(
            (spec_df['n_significant_clusters'] == 1).sum())
        top_novel = spec_df.head(20)[[
            'transcript_id', 'gene_id', 'class_code',
            'best_cluster', 'specificity_score'
        ]].to_dict('records')

    class_code_counts = {}
    if len(catalog) > 0:
        class_code_counts = catalog[
            'class_code'].value_counts().to_dict()

    summary = {
        'n_novel_in_annotations': n_novel,
        'n_novel_in_matrix': len(catalog),
        'n_enrichment_tests': len(enrichment_df),
        'n_significant_enrichments': n_sig,
        'n_cluster_specific_isoforms': n_cluster_specific,
        'class_code_distribution': class_code_counts,
        'top_novel_isoforms': top_novel,
    }

    with open(args.output_summary, 'w') as fh:
        json.dump(summary, fh, indent=2)
    logger.info(
        f"Novel isoform discovery complete: {len(catalog)} novel "
        f"isoforms, {n_sig} cluster-enriched, "
        f"{n_cluster_specific} cluster-specific.")
