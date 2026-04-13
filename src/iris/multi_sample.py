"""Cross-sample comparison of IRIS results.

Compares differential transcript usage, novel isoforms, cell type
composition, and isoform switching events across multiple samples
processed independently by IRIS. Identifies conserved and
sample-specific findings to support multi-sample experimental designs.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

from ._logging import get_named_logger, wf_parser

logger = get_named_logger("multisample")


def argparser():
    """Create argument parser."""
    parser = wf_parser("compare")

    parser.add_argument(
        "--sample_dirs", type=Path, nargs="+", required=True,
        help="List of IRIS output directories (one per sample).")
    parser.add_argument(
        "--sample_names", type=str, nargs="+", default=None,
        help="Optional sample names corresponding to sample_dirs. "
             "If omitted, directory names are used.")
    parser.add_argument(
        "--output_dir", type=Path, default="iris_comparison",
        help="Directory for comparison output files.")
    parser.add_argument(
        "--fdr_threshold", type=float, default=0.05,
        help="FDR threshold for significance calls.")

    return parser


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

_EXPECTED_FILES = {
    "dtu": "dtu_results.tsv",
    "switching": "isoform_switching.tsv",
    "novel_catalog": "novel_isoform_catalog.tsv",
    "novel_enrichment": "novel_isoform_summary.json",
    "clusters": "joint_clusters.tsv",
    "cell_types": "cell_type_annotations.tsv",
    "ase": "ase_results.tsv",
}


def load_sample_results(sample_dir, sample_name):
    """Load all available IRIS outputs from a single sample directory.

    Parameters
    ----------
    sample_dir : Path
        Path to the IRIS output directory for one sample.
    sample_name : str
        Human-readable name for this sample.

    Returns
    -------
    dict
        Keys are data categories (dtu, switching, novel_catalog,
        novel_enrichment, clusters, cell_types, ase). Values are
        DataFrames (or dict for JSON files) where the file exists,
        ``None`` otherwise. A ``'name'`` key holds the sample name.
    """
    sample_dir = Path(sample_dir)
    results = {"name": sample_name}

    for key, filename in _EXPECTED_FILES.items():
        filepath = sample_dir / filename
        if not filepath.exists():
            logger.info(
                "Sample '%s': %s not found, skipping.", sample_name, filename)
            results[key] = None
            continue

        if filename.endswith(".json"):
            with open(filepath) as fh:
                results[key] = json.load(fh)
        else:
            try:
                results[key] = pd.read_csv(filepath, sep="\t")
            except Exception as exc:
                logger.warning(
                    "Sample '%s': failed to load %s: %s",
                    sample_name, filename, exc)
                results[key] = None

    return results


# ---------------------------------------------------------------------------
# Comparison functions
# ---------------------------------------------------------------------------

def compare_dtu_across_samples(sample_results, fdr_threshold=0.05):
    """Compare DTU significance across samples.

    For each gene tested for DTU, determine which samples show
    significant differential transcript usage.

    Parameters
    ----------
    sample_results : list of dict
        Output from ``load_sample_results`` for each sample.
    fdr_threshold : float
        Significance threshold on adjusted p-values.

    Returns
    -------
    pd.DataFrame
        Columns: gene, n_samples_significant, conserved,
        sample_specific_in, plus per-sample p-value columns.
    """
    dtu_frames = {}
    for sr in sample_results:
        if sr["dtu"] is None:
            continue
        df = sr["dtu"].copy()
        # Identify columns
        gene_col = next(
            (c for c in df.columns if "gene" in c.lower()), df.columns[0])
        pv_col = next(
            (c for c in df.columns
             if "padj" in c.lower() or "pvalue_adj" in c.lower()
             or "fdr" in c.lower()),
            None)
        if pv_col is None:
            continue
        # Keep best (smallest) p-value per gene
        best = df.groupby(gene_col)[pv_col].min().reset_index()
        best.columns = ["gene", "pvalue_adj"]
        dtu_frames[sr["name"]] = best

    if not dtu_frames:
        logger.warning("No DTU results available for comparison.")
        return pd.DataFrame()

    # Merge all samples on gene
    all_genes = sorted(
        set(itertools.chain.from_iterable(
            f["gene"].tolist() for f in dtu_frames.values())))

    records = []
    for gene in all_genes:
        row = {"gene": gene}
        sig_samples = []
        for sname, df in dtu_frames.items():
            match = df.loc[df["gene"] == gene, "pvalue_adj"]
            pval = float(match.iloc[0]) if len(match) else np.nan
            row[f"pval_{sname}"] = pval
            if not np.isnan(pval) and pval < fdr_threshold:
                sig_samples.append(sname)
        n_tested = sum(
            1 for sname in dtu_frames
            if not np.isnan(row.get(f"pval_{sname}", np.nan)))
        row["n_samples_significant"] = len(sig_samples)
        row["conserved"] = (
            len(sig_samples) == n_tested and n_tested == len(dtu_frames))
        row["sample_specific_in"] = (
            sig_samples[0] if len(sig_samples) == 1 else "")
        records.append(row)

    result = pd.DataFrame(records)
    result = result.sort_values("n_samples_significant", ascending=False)
    return result.reset_index(drop=True)


def compare_novel_isoforms(sample_results):
    """Identify shared and sample-specific novel isoforms.

    Parameters
    ----------
    sample_results : list of dict
        Output from ``load_sample_results`` for each sample.

    Returns
    -------
    pd.DataFrame
        Columns: transcript_id, class_code, n_samples, samples_present,
        shared.
    """
    tx_records = {}  # transcript_id -> {class_code, samples}

    for sr in sample_results:
        if sr["novel_catalog"] is None:
            continue
        df = sr["novel_catalog"]
        tx_col = next(
            (c for c in df.columns if "transcript" in c.lower()), df.columns[0])
        cc_col = next(
            (c for c in df.columns if "class_code" in c.lower()), None)

        for _, row in df.iterrows():
            tid = row[tx_col]
            cc = row[cc_col] if cc_col else ""
            if tid not in tx_records:
                tx_records[tid] = {"class_code": cc, "samples": set()}
            tx_records[tid]["samples"].add(sr["name"])

    if not tx_records:
        logger.warning("No novel isoform catalogs available for comparison.")
        return pd.DataFrame()

    n_samples = len([s for s in sample_results if s["novel_catalog"] is not None])

    records = []
    for tid, info in tx_records.items():
        records.append({
            "transcript_id": tid,
            "class_code": info["class_code"],
            "n_samples": len(info["samples"]),
            "samples_present": ",".join(sorted(info["samples"])),
            "shared": len(info["samples"]) == n_samples,
        })

    result = pd.DataFrame(records)
    result = result.sort_values(
        ["n_samples", "transcript_id"], ascending=[False, True])
    return result.reset_index(drop=True)


def compare_cell_type_composition(sample_results):
    """Compare cell type proportions across samples.

    For each pair of samples, a Fisher's exact test is run per cell type
    to detect differential composition.

    Parameters
    ----------
    sample_results : list of dict
        Output from ``load_sample_results`` for each sample.

    Returns
    -------
    pd.DataFrame
        Composition matrix with per-cell-type proportions and
        pair-wise Fisher's exact test p-values appended.
    """
    composition = {}  # sample_name -> {cell_type: count}

    for sr in sample_results:
        if sr["cell_types"] is None:
            continue
        df = sr["cell_types"]
        ct_col = next(
            (c for c in df.columns if "cell_type" in c.lower()), df.columns[-1])
        counts = df[ct_col].value_counts().to_dict()
        composition[sr["name"]] = counts

    if len(composition) < 2:
        logger.warning(
            "Need at least 2 samples with cell type annotations for "
            "composition comparison.")
        return pd.DataFrame()

    all_types = sorted(
        set(itertools.chain.from_iterable(composition.values())))
    sample_names = sorted(composition.keys())

    # Build proportion matrix
    rows = []
    for sname in sample_names:
        total = sum(composition[sname].values())
        row = {"sample": sname, "_total": total}
        for ct in all_types:
            cnt = composition[sname].get(ct, 0)
            row[f"{ct}_count"] = cnt
            row[f"{ct}_proportion"] = cnt / total if total else 0.0
        rows.append(row)
    comp_df = pd.DataFrame(rows)

    # Pairwise Fisher's exact test per cell type
    fisher_records = []
    for ct in all_types:
        for s1, s2 in itertools.combinations(sample_names, 2):
            a = composition[s1].get(ct, 0)
            b = sum(composition[s1].values()) - a
            c = composition[s2].get(ct, 0)
            d = sum(composition[s2].values()) - c
            try:
                _, pval = scipy.stats.fisher_exact([[a, b], [c, d]])
            except Exception:
                pval = np.nan
            fisher_records.append({
                "cell_type": ct,
                "sample_1": s1,
                "sample_2": s2,
                "count_s1": a,
                "count_s2": c,
                "pvalue": pval,
            })

    fisher_df = pd.DataFrame(fisher_records)
    # BH correction
    if len(fisher_df) > 0 and fisher_df["pvalue"].notna().any():
        pvals = fisher_df["pvalue"].fillna(1.0).values
        n = len(pvals)
        ranked = np.argsort(pvals)
        adjusted = np.ones(n)
        for i, rank_idx in enumerate(ranked):
            adjusted[rank_idx] = pvals[rank_idx] * n / (i + 1)
        adjusted = np.minimum.accumulate(adjusted[np.argsort(ranked)][::-1])[::-1]
        adjusted = np.clip(adjusted, 0, 1)
        fisher_df["pvalue_adj"] = adjusted[np.argsort(np.argsort(pvals))]

    # Combine into single output: composition rows then fisher rows
    comp_df.drop(columns=["_total"], inplace=True)

    return comp_df, fisher_df


def compare_switching_events(sample_results):
    """Identify conserved and sample-specific isoform switching events.

    A switching event is considered conserved when the same gene shows
    the same dominant transcript change across samples.

    Parameters
    ----------
    sample_results : list of dict
        Output from ``load_sample_results`` for each sample.

    Returns
    -------
    pd.DataFrame
        Columns: gene, transcript_1, transcript_2, n_samples,
        samples_present, conserved.
    """
    switch_records = {}  # (gene, tx1, tx2) -> set of samples

    for sr in sample_results:
        if sr["switching"] is None:
            continue
        df = sr["switching"]
        gene_col = next(
            (c for c in df.columns if "gene" in c.lower()), df.columns[0])
        tx_cols = [c for c in df.columns if "transcript" in c.lower()]

        if len(tx_cols) < 2:
            # Try to find dominant isoform columns
            tx_cols = [c for c in df.columns
                       if "dominant" in c.lower() or "isoform" in c.lower()]

        if len(tx_cols) < 2:
            logger.info(
                "Sample '%s': cannot identify transcript pair columns in "
                "isoform_switching.tsv, skipping switching comparison.",
                sr["name"])
            continue

        tx1_col, tx2_col = tx_cols[0], tx_cols[1]

        for _, row in df.iterrows():
            gene = row[gene_col]
            t1 = str(row[tx1_col])
            t2 = str(row[tx2_col])
            # Canonical ordering
            key = (gene, min(t1, t2), max(t1, t2))
            if key not in switch_records:
                switch_records[key] = set()
            switch_records[key].add(sr["name"])

    if not switch_records:
        logger.warning("No switching results available for comparison.")
        return pd.DataFrame()

    n_samples_with_data = len(
        [s for s in sample_results if s["switching"] is not None])

    records = []
    for (gene, t1, t2), samples in switch_records.items():
        records.append({
            "gene": gene,
            "transcript_1": t1,
            "transcript_2": t2,
            "n_samples": len(samples),
            "samples_present": ",".join(sorted(samples)),
            "conserved": len(samples) == n_samples_with_data,
        })

    result = pd.DataFrame(records)
    result = result.sort_values(
        ["n_samples", "gene"], ascending=[False, True])
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(args):
    """Run all cross-sample comparisons and write output files.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments with sample_dirs, sample_names,
        output_dir, and fdr_threshold fields.
    """
    logger.info("IRIS Multi-Sample Comparison")

    sample_dirs = [Path(d) for d in args.sample_dirs]
    sample_names = args.sample_names
    if sample_names is None:
        sample_names = [d.name for d in sample_dirs]
    if len(sample_names) != len(sample_dirs):
        raise ValueError(
            f"Number of sample names ({len(sample_names)}) does not match "
            f"number of sample directories ({len(sample_dirs)}).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fdr = args.fdr_threshold

    logger.info(
        "Comparing %d samples: %s", len(sample_names),
        ", ".join(sample_names))

    # Load all samples
    all_results = []
    for sdir, sname in zip(sample_dirs, sample_names):
        if not sdir.exists():
            logger.warning("Sample directory does not exist: %s", sdir)
            continue
        sr = load_sample_results(sdir, sname)
        all_results.append(sr)

    if len(all_results) < 2:
        logger.error(
            "Need at least 2 valid sample directories. Found %d.",
            len(all_results))
        return

    summary = {"n_samples": len(all_results), "sample_names": sample_names}

    # --- DTU comparison ---
    dtu_comp = compare_dtu_across_samples(all_results, fdr_threshold=fdr)
    if len(dtu_comp):
        conserved = dtu_comp[dtu_comp["conserved"]]
        sample_specific = dtu_comp[dtu_comp["sample_specific_in"] != ""]

        conserved.to_csv(
            output_dir / "conserved_dtu.tsv", sep="\t", index=False)
        sample_specific.to_csv(
            output_dir / "sample_specific_dtu.tsv", sep="\t", index=False)

        summary["dtu"] = {
            "total_genes_tested": len(dtu_comp),
            "conserved_dtu_genes": int(conserved.shape[0]),
            "sample_specific_dtu_genes": int(sample_specific.shape[0]),
        }
        logger.info(
            "DTU: %d conserved, %d sample-specific genes",
            len(conserved), len(sample_specific))
    else:
        summary["dtu"] = {"total_genes_tested": 0}

    # --- Novel isoform comparison ---
    novel_comp = compare_novel_isoforms(all_results)
    if len(novel_comp):
        novel_comp.to_csv(
            output_dir / "shared_novel_isoforms.tsv", sep="\t", index=False)
        n_shared = int(novel_comp["shared"].sum())
        n_total = len(novel_comp)
        summary["novel_isoforms"] = {
            "total_novel": n_total,
            "shared_across_all": n_shared,
            "sample_specific": n_total - n_shared,
        }
        logger.info(
            "Novel isoforms: %d total, %d shared across all samples",
            n_total, n_shared)
    else:
        summary["novel_isoforms"] = {"total_novel": 0}

    # --- Cell type composition ---
    comp_result = compare_cell_type_composition(all_results)
    if isinstance(comp_result, tuple) and len(comp_result) == 2:
        comp_df, fisher_df = comp_result
        comp_df.to_csv(
            output_dir / "composition_comparison.tsv", sep="\t", index=False)
        if len(fisher_df):
            fisher_df.to_csv(
                output_dir / "composition_fisher_tests.tsv",
                sep="\t", index=False)
        sig_ct = 0
        if "pvalue_adj" in fisher_df.columns:
            sig_ct = int((fisher_df["pvalue_adj"] < fdr).sum())
        summary["cell_type_composition"] = {
            "n_cell_types": len([
                c for c in comp_df.columns if c.endswith("_proportion")]),
            "significant_differences": sig_ct,
        }
        logger.info(
            "Cell type composition: %d significant differences (FDR < %g)",
            sig_ct, fdr)

    # --- Switching comparison ---
    switch_comp = compare_switching_events(all_results)
    if len(switch_comp):
        switch_comp.to_csv(
            output_dir / "conserved_switches.tsv", sep="\t", index=False)
        n_conserved_sw = int(switch_comp["conserved"].sum())
        summary["switching"] = {
            "total_events": len(switch_comp),
            "conserved_switches": n_conserved_sw,
        }
        logger.info(
            "Switching: %d total events, %d conserved across all samples",
            len(switch_comp), n_conserved_sw)
    else:
        summary["switching"] = {"total_events": 0}

    # --- Write summary ---
    with open(output_dir / "comparison_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info(
        "Comparison complete. Results written to: %s", output_dir.resolve())
