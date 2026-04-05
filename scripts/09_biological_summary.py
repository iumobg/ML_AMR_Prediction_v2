#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Biological Summary Report Generator — Step 09

Generates a Markdown report (05_final_biological_report.md) that maps the
top k-mer features (from Step 07) to their biological meaning via:

  1. CARD local BLAST results  → acquired resistance gene names
  2. NCBI remote BLAST results → core-genome / SNP context

For NCBI hits, instead of using the generic 'stitle' (which typically says
"complete genome"), this script queries NCBI Entrez efetch in real-time to
retrieve the specific gene/product name that overlaps the matched coordinates.

API behaviour is throttled (0.3 s between calls) and fully wrapped in
try/except so the script never crashes from network errors.
"""

# ============================================================================
# LIBRARY IMPORTS
# ============================================================================
import sys
import re
import time
import yaml
import pandas as pd
from pathlib import Path
from Bio import Entrez, SeqIO

# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# NCBI requires a registered e-mail for Entrez queries.
Entrez.email = "user@example.com"


# ============================================================================
# CONFIGURATION LOADER
# ============================================================================
def load_config(config_path=None):
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)


# ============================================================================
# CARD HELPER
# ============================================================================
def extract_card_gene(sseqid):
    """Extract AMR gene symbol from CARD sseqid.

    Example:
        gb|NG_068181.1|+|100-925|ARO:3006096|OXA-909  →  OXA-909
    """
    sseqid = str(sseqid)
    if '|' in sseqid:
        return sseqid.split('|')[-1].strip()
    return sseqid


# ============================================================================
# NCBI STITLE CLEANER  (used as fallback when Entrez lookup fails)
# ============================================================================
def clean_ncbi_stitle(stitle):
    """Strip generic genome-level metadata from an NCBI stitle string."""
    stitle = str(stitle)
    patterns = [
        r",\s*complete genome",
        r"\s*complete genome",
        r",\s*complete sequence",
        r"\s*complete sequence",
        r"\s*genome assembly,\s*chromosome:\s*\w+",
        r"\s*genome assembly,\s*chromosome",
        r"\s*genome assembly.*",
        r"\s*chromosome,.*",
        r"\s*chromosome.*",
        r"\s*plasmid.*",
        r"\s+DNA,.*",
        r"\s+DNA.*",
        r",\s*partial cds",
        r"\s*partial cds",
        r"\s*gene for.*",
    ]
    cleaned = stitle
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


# ============================================================================
# ENTREZ COORDINATE-BASED GENE NAME LOOKUP
# ============================================================================
def _extract_accession(sseqid: str) -> str:
    """Return the bare accession number from a raw BLAST sseqid field.

    Handles formats such as:
        gi|123456|gb|NZ_CP012345.1|    →  NZ_CP012345.1
        ref|NZ_CP012345.1|             →  NZ_CP012345.1
        NZ_CP012345.1                  →  NZ_CP012345.1
    """
    sseqid = str(sseqid).strip()
    if '|' in sseqid:
        parts = [p for p in sseqid.split('|') if p.strip()]
        # The accession is the last non-empty token
        return parts[-1].strip()
    return sseqid


def fetch_gene_name_at_coords(sseqid: str, sstart: int, send: int,
                               stitle: str) -> str:
    """Query NCBI Entrez to find the gene/product overlapping [sstart, send].

    Parameters
    ----------
    sseqid  : raw BLAST subject sequence ID (accession, possibly pipe-delimited)
    sstart  : subject alignment start (1-based)
    send    : subject alignment end   (1-based)
    stitle  : original BLAST stitle column (used for fallback organism label)

    Returns
    -------
    A human-readable string in one of three formats:
        "GeneName (OrganismName)"          – successful Entrez lookup
        "Intergenic Region (OrganismName)" – no CDS/gene feature at coords
        "API Error (OrganismName)"         – network / parse error
    """
    organism_label = clean_ncbi_stitle(stitle)
    accession = _extract_accession(sseqid)

    # Reverse-strand hits have sstart > send; normalise for efetch
    seq_start = min(sstart, send)
    seq_stop  = max(sstart, send)

    try:
        handle = Entrez.efetch(
            db="nucleotide",
            id=accession,
            rettype="gb",
            retmode="text",
            seq_start=seq_start,
            seq_stop=seq_stop,
        )
        record = SeqIO.read(handle, "genbank")
        handle.close()

        # Walk features looking for an annotated gene or product qualifier
        for feature in record.features:
            if feature.type not in ("CDS", "gene", "rRNA", "tRNA", "ncRNA",
                                    "misc_RNA", "misc_feature"):
                continue

            qualifiers = feature.qualifiers

            # Prefer /gene over /product (shorter, canonical symbol)
            gene_name = (
                qualifiers.get("gene",    [None])[0]
                or qualifiers.get("product", [None])[0]
            )
            if gene_name:
                return f"{gene_name} ({organism_label})"

        # No annotated feature found at these coordinates
        return f"Intergenic Region ({organism_label})"

    except Exception:
        return f"API Error ({organism_label})"


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("Loading configuration...")
    config = load_config()
    antibiotic = config['project']['target_antibiotic']

    # ------------------------------------------------------------------
    # Resolve paths from config
    # ------------------------------------------------------------------
    explain_dir_template = config['paths']['dir_05_explainability']
    if "{antibiotic}" in explain_dir_template:
        explain_dir_path = explain_dir_template.format(antibiotic=antibiotic)
    else:
        explain_dir_path = explain_dir_template

    explain_dir = PROJECT_ROOT / explain_dir_path
    if not explain_dir.exists():
        print(f"Error: Directory {explain_dir} does not exist.")
        sys.exit(1)

    csv_file  = explain_dir / f"01_top_50_features_{antibiotic}.csv"
    card_file = explain_dir / f"03_card_blast_results_{antibiotic}.tsv"
    ncbi_file = explain_dir / f"04_ncbi_blast_results_{antibiotic}.tsv"
    out_file  = explain_dir / "05_final_biological_report.md"

    if not csv_file.exists():
        print(f"Error: Cannot find top features CSV at {csv_file}")
        sys.exit(1)

    print(f"Reading {csv_file}...")
    df_features = pd.read_csv(csv_file)

    # ------------------------------------------------------------------
    # TSV column schema shared by both BLAST result files
    # ------------------------------------------------------------------
    tsv_cols = [
        'qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen',
        'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore', 'stitle',
    ]

    # ------------------------------------------------------------------
    # Load & filter CARD results  (gene names extracted offline from sseqid)
    # ------------------------------------------------------------------
    if card_file.exists() and card_file.stat().st_size > 0:
        print(f"Reading {card_file}...")
        df_card = pd.read_csv(card_file, sep='\t', header=None, names=tsv_cols)
        df_card['pident'] = pd.to_numeric(df_card['pident'], errors='coerce')
        df_card['evalue'] = pd.to_numeric(df_card['evalue'], errors='coerce')
        df_card = df_card[
            (df_card['pident'] >= 90.0) & (df_card['evalue'] <= 50.0)
        ].copy()
        df_card['Gene_Match'] = df_card['sseqid'].apply(extract_card_gene)
    else:
        print(f"Warning: {card_file} is missing or empty.")
        df_card = pd.DataFrame(columns=tsv_cols + ['Gene_Match'])

    # ------------------------------------------------------------------
    # Load & filter NCBI results  (gene names resolved at report-write time)
    # Gene_Match column is intentionally left blank here; it will be
    # populated row-by-row inside the report loop below.
    # ------------------------------------------------------------------
    if ncbi_file.exists() and ncbi_file.stat().st_size > 0:
        print(f"Reading {ncbi_file}...")
        df_ncbi = pd.read_csv(ncbi_file, sep='\t', header=None, names=tsv_cols)
        df_ncbi['pident'] = pd.to_numeric(df_ncbi['pident'], errors='coerce')
        df_ncbi['evalue'] = pd.to_numeric(df_ncbi['evalue'], errors='coerce')
        df_ncbi['sstart'] = pd.to_numeric(df_ncbi['sstart'], errors='coerce').fillna(0).astype(int)
        df_ncbi['send']   = pd.to_numeric(df_ncbi['send'],   errors='coerce').fillna(0).astype(int)
        df_ncbi = df_ncbi[
            (df_ncbi['pident'] >= 90.0) & (df_ncbi['evalue'] <= 50.0)
        ].copy()
    else:
        print(f"Warning: {ncbi_file} is missing or empty.")
        df_ncbi = pd.DataFrame(columns=tsv_cols)

    # ------------------------------------------------------------------
    # Generate Markdown report
    # ------------------------------------------------------------------
    print(f"Generating markdown report at {out_file}...")

    with open(out_file, "w") as f:
        f.write("# Final Biological Report\n")
        f.write(f"**Target Antibiotic:** {antibiotic.capitalize()}\n\n")
        f.write("---\n\n")

        for _, row in df_features.iterrows():
            rank     = int(row['Rank'])
            score    = float(row['Gain_Score'])
            feat_id  = str(row['Feature_ID'])
            sequence = str(row['Kmer_Sequence'])

            # Reconstruct the query ID to match BLAST qseqid column
            q_id = f"Rank_{rank}|Score_{score:.4f}|Feature_{feat_id}"

            f.write(f"### Rank {rank}: {sequence} (Gain: {score:.4f})\n")

            # --------------------------------------------------------------
            # CARD hits — no Entrez call needed, offline gene symbol lookup
            # --------------------------------------------------------------
            f.write("**CARD Hits (Acquired Resistance / Plasmids):**\n")
            card_hits = df_card[df_card['qseqid'] == q_id].head(10)
            if not card_hits.empty:
                for _, hit in card_hits.iterrows():
                    f.write(
                        f"- {hit['Gene_Match']}, "
                        f"Identity: {hit['pident']}%, "
                        f"E-value: {hit['evalue']}\n"
                    )
            else:
                f.write("*No high-confidence hits*\n")

            # --------------------------------------------------------------
            # NCBI hits — real-time Entrez coordinate lookup (top 10 only)
            # --------------------------------------------------------------
            f.write("**NCBI Hits (Core Genome / SNPs):**\n")
            ncbi_hits = df_ncbi[df_ncbi['qseqid'] == q_id].head(10)
            if not ncbi_hits.empty:
                for _, hit in ncbi_hits.iterrows():
                    gene_label = fetch_gene_name_at_coords(
                        sseqid=hit['sseqid'],
                        sstart=int(hit['sstart']),
                        send=int(hit['send']),
                        stitle=hit['stitle'],
                    )
                    f.write(
                        f"- {gene_label}, "
                        f"Identity: {hit['pident']}%, "
                        f"E-value: {hit['evalue']}\n"
                    )
                    # Be polite to the NCBI API
                    time.sleep(0.3)
            else:
                f.write("*No high-confidence hits*\n")

            f.write("\n")

    print("Generation complete!")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()
