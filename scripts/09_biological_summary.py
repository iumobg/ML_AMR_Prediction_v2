#!/usr/bin/env python3
import pandas as pd
import yaml
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_config(config_path=None):
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

def extract_card_gene(sseqid):
    """Extract AMR gene symbol from CARD sseqid."""
    sseqid = str(sseqid)
    # e.g., gb|NG_068181.1|+|100-925|ARO:3006096|OXA-909 -> OXA-909
    if '|' in sseqid:
        return sseqid.split('|')[-1].strip()
    return sseqid

def clean_ncbi_stitle(stitle):
    """Clean generic metadata out of NCBI stitle."""
    stitle = str(stitle)
    # Generic terms to remove (case insensitive)
    to_remove = [
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
    for pattern in to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

def main():
    print("Loading configuration...")
    config = load_config()
    antibiotic = config['project']['target_antibiotic']
    
    # Paths
    explain_dir_template = config['paths']['dir_05_explainability']
    if "{antibiotic}" in explain_dir_template:
        explain_dir_path = explain_dir_template.format(antibiotic=antibiotic)
    else:
        explain_dir_path = explain_dir_template

    explain_dir = PROJECT_ROOT / explain_dir_path
    if not explain_dir.exists():
        print(f"Error: Directory {explain_dir} does not exist.")
        sys.exit(1)
        
    csv_file = explain_dir / f"01_top_50_features_{antibiotic}.csv"
    card_file = explain_dir / f"03_card_blast_results_{antibiotic}.tsv"
    ncbi_file = explain_dir / f"04_ncbi_blast_results_{antibiotic}.tsv"
    out_file = explain_dir / "05_final_biological_report.md"
    
    if not csv_file.exists():
        print(f"Error: Cannot find top features CSV at {csv_file}")
        sys.exit(1)
        
    print(f"Reading {csv_file}...")
    df_features = pd.read_csv(csv_file)
    
    # TSV Columns
    tsv_cols = ['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore', 'stitle']
    
    # Load and filter CARD
    if card_file.exists() and card_file.stat().st_size > 0:
        print(f"Reading {card_file}...")
        df_card = pd.read_csv(card_file, sep='\t', header=None, names=tsv_cols)
        df_card = df_card[(df_card['pident'] >= 90.0) & (df_card['evalue'] <= 50.0)].copy()
        df_card['Gene_Match'] = df_card['sseqid'].apply(extract_card_gene)
    else:
        print(f"Warning: {card_file} is missing or empty.")
        df_card = pd.DataFrame(columns=tsv_cols + ['Gene_Match'])

    # Load and filter NCBI
    if ncbi_file.exists() and ncbi_file.stat().st_size > 0:
        print(f"Reading {ncbi_file}...")
        # Since NCBI can be large, we'll read it straight
        df_ncbi = pd.read_csv(ncbi_file, sep='\t', header=None, names=tsv_cols)
        df_ncbi = df_ncbi[(df_ncbi['pident'] >= 90.0) & (df_ncbi['evalue'] <= 50.0)].copy()
        df_ncbi['Gene_Match'] = df_ncbi['stitle'].apply(clean_ncbi_stitle)
    else:
        print(f"Warning: {ncbi_file} is missing or empty.")
        df_ncbi = pd.DataFrame(columns=tsv_cols + ['Gene_Match'])

    print(f"Generating markdown report at {out_file}...")
    with open(out_file, "w") as f:
        f.write("# Final Biological Report\n")
        f.write(f"**Target Antibiotic:** {antibiotic.capitalize()}\n\n")
        f.write("---\n\n")
        
        for _, row in df_features.iterrows():
            rank = int(row['Rank'])
            score = float(row['Gain_Score'])
            feat_id = str(row['Feature_ID'])
            sequence = str(row['Kmer_Sequence'])
            
            # Exact reconstruction of q_id
            q_id = f"Rank_{rank}|Score_{score:.4f}|Feature_{feat_id}"
            
            f.write(f"### Rank {rank}: {sequence} (Gain: {score:.4f})\n")
            
            # CARD Hits
            f.write("**CARD Hits (Acquired Resistance / Plasmids):**\n")
            card_hits = df_card[df_card['qseqid'] == q_id].head(5)
            if not card_hits.empty:
                for _, hit in card_hits.iterrows():
                    f.write(f"- {hit['Gene_Match']}, Identity: {hit['pident']}%, E-value: {hit['evalue']}\n")
            else:
                f.write("*No high-confidence hits*\n")
                
            # NCBI Hits
            f.write("**NCBI Hits (Core Genome / SNPs):**\n")
            ncbi_hits = df_ncbi[df_ncbi['qseqid'] == q_id].head(5)
            if not ncbi_hits.empty:
                for _, hit in ncbi_hits.iterrows():
                    f.write(f"- {hit['Gene_Match']}, Identity: {hit['pident']}%, E-value: {hit['evalue']}\n")
            else:
                f.write("*No high-confidence hits*\n")
            
            f.write("\n")
            
    print("Generation complete!")

if __name__ == "__main__":
    main()
