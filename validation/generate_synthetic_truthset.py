#!/usr/bin/env python3
"""
Generate synthetic validation data for IRIS's novel isoform discovery and ASE modules.

Produces:
  1. synthetic_annotations.gtf  - gffcompare-annotated GTF with class codes
  2. synthetic_variants.vcf     - heterozygous SNV VCF
  3. synthetic_tagged.bam       - CB-tagged BAM covering variant positions
"""

import gzip
import os
import random
import string

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Load real IDs from the existing validation data
# ---------------------------------------------------------------------------

def load_transcript_gene_map(path):
    """Read gene_transcript_map.tsv -> list of (transcript_id, gene_id)."""
    pairs = []
    with open(path) as fh:
        header = next(fh)
        for line in fh:
            tx, gene = line.strip().split("\t")
            pairs.append((tx, gene))
    return pairs

def load_barcodes(path):
    """Read barcodes.tsv.gz -> list of barcode strings."""
    barcodes = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            barcodes.append(line.strip())
    return barcodes

tx_gene_pairs = load_transcript_gene_map(os.path.join(BASE_DIR, "gene_transcript_map.tsv"))
barcodes = load_barcodes(os.path.join(BASE_DIR, "gene_matrix", "barcodes.tsv.gz"))

print(f"Loaded {len(tx_gene_pairs)} transcript-gene pairs and {len(barcodes)} barcodes.")

# ---------------------------------------------------------------------------
# Chromosome helpers
# ---------------------------------------------------------------------------

CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
CHROM_LENGTHS = {c: 50_000_000 + random.randint(0, 200_000_000) for c in CHROMS}

BASES = ["A", "C", "G", "T"]

def random_alt(ref):
    return random.choice([b for b in BASES if b != ref])

# ---------------------------------------------------------------------------
# 1. Generate synthetic_annotations.gtf
# ---------------------------------------------------------------------------

CLASS_CODE_DIST = {
    "=": 0.60,  # known
    "j": 0.25,  # novel splice variant
    "u": 0.10,  # novel intergenic
    "x": 0.05,  # antisense
}

def assign_class_codes(n):
    codes = []
    for cc, frac in CLASS_CODE_DIST.items():
        codes.extend([cc] * int(round(n * frac)))
    # fill or trim to exact size
    while len(codes) < n:
        codes.append("=")
    codes = codes[:n]
    random.shuffle(codes)
    return codes

def generate_gtf(path, tx_gene_pairs):
    n = len(tx_gene_pairs)
    codes = assign_class_codes(n)

    with open(path, "w") as fh:
        fh.write("##gtf-version 2\n")
        fh.write("##generated-by generate_synthetic_truthset.py\n")
        for i, (tx_id, gene_id) in enumerate(tx_gene_pairs):
            chrom = CHROMS[i % len(CHROMS)]
            strand = "+" if random.random() < 0.5 else "-"
            start = random.randint(1000, CHROM_LENGTHS[chrom] - 10000)
            end = start + random.randint(500, 5000)
            cc = codes[i]

            # For novel intergenic transcripts, ref fields are "NA"
            if cc == "u":
                ref_gene = "NA"
                ref_tx = "NA"
            else:
                ref_gene = gene_id
                ref_tx = tx_id

            attrs = (
                f'transcript_id "{tx_id}"; '
                f'gene_id "{gene_id}"; '
                f'class_code "{cc}"; '
                f'ref_gene_id "{ref_gene}"; '
                f'cmp_ref "{ref_tx}";'
            )
            fh.write(f"{chrom}\tgffcompare\ttranscript\t{start}\t{end}\t.\t{strand}\t.\t{attrs}\n")

    # Report class code counts
    from collections import Counter
    counts = Counter(codes)
    print(f"GTF: wrote {n} transcript entries to {path}")
    for cc in ["=", "j", "u", "x"]:
        print(f"  class_code '{cc}': {counts.get(cc, 0)}  ({counts.get(cc, 0)/n*100:.1f}%)")

gtf_path = os.path.join(BASE_DIR, "synthetic_annotations.gtf")
generate_gtf(gtf_path, tx_gene_pairs)

# ---------------------------------------------------------------------------
# 2. Generate synthetic_variants.vcf
# ---------------------------------------------------------------------------

NUM_VARIANTS = 200

def generate_vcf(path):
    variants = []
    for _ in range(NUM_VARIANTS):
        chrom = random.choice(CHROMS)
        pos = random.randint(10000, CHROM_LENGTHS[chrom])
        ref = random.choice(BASES)
        alt = random_alt(ref)
        qual = random.randint(20, 60)
        variants.append((chrom, pos, ref, alt, qual))

    # Sort by chrom then pos for a well-formed VCF
    chrom_order = {c: i for i, c in enumerate(CHROMS)}
    variants.sort(key=lambda v: (chrom_order[v[0]], v[1]))

    with open(path, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
        for chrom, pos, ref, alt, qual in variants:
            fh.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t{qual}\tPASS\t.\tGT\t0/1\n")

    print(f"VCF: wrote {NUM_VARIANTS} heterozygous variants to {path}")
    return variants

vcf_path = os.path.join(BASE_DIR, "synthetic_variants.vcf")
variants = generate_vcf(vcf_path)

# ---------------------------------------------------------------------------
# 3. Generate synthetic_tagged.bam
# ---------------------------------------------------------------------------

NUM_READS = 500
READ_LEN = 150

def generate_bam(path, variants, barcodes):
    try:
        import pysam
    except ImportError:
        print("BAM: pysam not available -- skipping BAM generation.")
        return

    # Build a minimal header with all chromosomes
    sq_entries = [{"SN": c, "LN": CHROM_LENGTHS[c]} for c in CHROMS]
    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": sq_entries,
        "RG": [{"ID": "synthetic", "SM": "SAMPLE"}],
    })

    chrom_to_tid = {c: i for i, c in enumerate(CHROMS)}

    reads = []
    for i in range(NUM_READS):
        # Pick a variant position to anchor the read near
        chrom, vpos, ref, alt, _ = random.choice(variants)

        # Place the read so it covers the variant
        offset = random.randint(0, READ_LEN - 1)
        start = max(0, vpos - offset - 1)  # 0-based

        a = pysam.AlignedSegment(header)
        a.query_name = f"read_{i:05d}"
        a.query_sequence = "".join(random.choices(BASES, k=READ_LEN))
        a.flag = 0
        a.reference_id = chrom_to_tid[chrom]
        a.reference_start = start
        a.mapping_quality = random.randint(20, 60)
        a.cigar = [(0, READ_LEN)]  # simple M alignment
        a.query_qualities = pysam.qualitystring_to_array(
            "".join(chr(random.randint(33 + 20, 33 + 40)) for _ in range(READ_LEN))
        )

        # Cell barcode tag
        a.set_tag("CB", random.choice(barcodes), value_type="Z")
        a.set_tag("RG", "synthetic", value_type="Z")

        reads.append(a)

    # Sort by coordinate
    reads.sort(key=lambda r: (r.reference_id, r.reference_start))

    # Write BAM
    with pysam.AlignmentFile(path, "wb", header=header) as outf:
        for a in reads:
            outf.write(a)

    # Index
    pysam.sort("-o", path, path)
    pysam.index(path)

    print(f"BAM: wrote {NUM_READS} reads to {path}")
    print(f"BAM: index created at {path}.bai")

bam_path = os.path.join(BASE_DIR, "synthetic_tagged.bam")
generate_bam(bam_path, variants, barcodes)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print("\n--- Synthetic truth-set generation complete ---")
for f in ["synthetic_annotations.gtf", "synthetic_variants.vcf",
          "synthetic_tagged.bam", "synthetic_tagged.bam.bai"]:
    fp = os.path.join(BASE_DIR, f)
    if os.path.exists(fp):
        size_kb = os.path.getsize(fp) / 1024
        print(f"  {f}: {size_kb:.1f} KB")
    else:
        print(f"  {f}: NOT CREATED")
