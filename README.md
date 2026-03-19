# ML_AMR_Prediction_v2

> **A scalable, memory-efficient machine learning pipeline for Antimicrobial Resistance (AMR) prediction from Whole-Genome Sequencing (WGS) data using k-mer features and an optimized, out-of-core XGBoost model.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-orange.svg)](https://xgboost.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-2.10+-green.svg)](https://optuna.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Antimicrobial resistance (AMR) is a critical global health threat. This pipeline predicts AMR phenotype directly from bacterial whole-genome sequencing (WGS) data — **without requiring sequence alignment or prior knowledge of resistance genes**. Instead, it leverages the raw discriminative power of k-mer frequencies: short DNA subsequences whose presence or absence is directly linked to genetic mutations conferring resistance.

The pipeline is designed to handle the extreme scale of genomic data:
- **Millions of features** (21-mer space: ~4.4 × 10¹²  theoretical, tens of millions observed)  
- **Sparse, high-dimensional matrices** stored in compressed `.npz` format  
- **Out-of-core learning** via chunked XGBoost training to stay within RAM limits  
- **Principled hyperparameter optimization** via Optuna with biologically-motivated feature subsampling

---

## Pipeline Architecture

The pipeline consists of five sequential scripts, each handling a distinct phase of the workflow.

```
WGS FASTA Files
      │
      ▼
┌─────────────────────────────┐
│  01_data_validation.py      │  Quality control & metadata validation
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  02_kmer_extraction.py      │  KMC3-based k-mer counting (k=21)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  03_matrix_construction.py  │  Sparse CSR matrix assembly (chunked .npz)
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  04_optimization.py         │  Optuna HPO with smart chunk selection
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  05_model_training.py       │  Epoch-based incremental XGBoost training
└─────────────────────────────┘
```

### Step-by-Step Description

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_data_validation.py` | Validates raw FASTA files for completeness, sequence integrity, and metadata consistency. Flags corrupted or duplicate genomes and outputs a clean manifest for downstream processing. |
| 2 | `02_kmer_extraction.py` | Invokes **KMC v3** to count all canonical 21-mers within each genome. Outputs per-genome k-mer databases, globally intersects them to build a shared vocabulary, and converts to binary presence/absence vectors. |
| 3 | `03_matrix_construction.py` | Assembles individual genome vectors into a global sparse matrix (SciPy CSR format), chunked across disk to avoid loading the full dataset into RAM. Applies prevalence filtering to remove uninformative k-mers (present in all or no genomes). |
| 4 | `04_optimization.py` | Runs a **Bayesian hyperparameter search** using Optuna. Uses the **Square Root Heuristic** (`colsample_bytree` ≈ √p / p) for feature subsampling and the **stratified linspace** chunk selection strategy to maintain resistance ratio balance across mini-batches. Early stopping determines the optimal `n_estimators` per trial. |
| 5 | `05_model_training.py` | Trains the final XGBoost model using the best configuration from Step 4. Employs **epoch-based incremental learning**: shuffling chunks across multiple epochs to prevent catastrophic forgetting and ensure the model learns from all genomic data segments uniformly. |

---

## Installation

### Prerequisites

- Python ≥ 3.8
- [KMC v3.2+](https://github.com/refresh-bio/KMC/releases) — k-mer counting engine (must be accessible in `bin/`)
- Conda or virtualenv (recommended)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ML_AMR_Prediction_v2.git
cd ML_AMR_Prediction_v2

# 2. Create and activate a virtual environment
conda create -n amr_env python=3.10 -y
conda activate amr_env

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Place KMC binaries
# Download from: https://github.com/refresh-bio/KMC/releases
# Extract to:    bin/kmc and bin/kmc_tools
chmod +x bin/kmc bin/kmc_tools
```

---

## Configuration

All pipeline parameters are centralized in `config/config.yaml`. Antibiotic-specific parameters (set by the optimizer) are saved to `config/config_<antibiotic>.yaml`.

Key parameters to review before running:

```yaml
# config/config.yaml (excerpt)
kmer_size: 21                   # k-mer length (biological default: 21)
antibiotic: ciprofloxacin       # Target antibiotic for AMR prediction
data:
  raw_genomes_dir: data/raw_genomes/
  matrix_chunks_dir: data/ciprofloxacin/
model:
  max_bin: 2                    # Binary features → 1-bit histograms (RAM saver)
  n_epochs: 3                   # Training epochs over all chunks
```

---

## Usage

Run each script in order from the project root:

```bash
# Step 1: Validate raw genomic data
python scripts/01_data_validation.py

# Step 2: Extract k-mers using KMC
python scripts/02_kmer_extraction.py

# Step 3: Build sparse feature matrices
python scripts/03_matrix_construction.py

# Step 4: Optimize hyperparameters with Optuna
python scripts/04_optimization.py

# Step 5: Train the final model
python scripts/05_model_training.py
```

Monitor progress via logs in `logs/<antibiotic>/`.

---

## Project Structure

```
ML_AMR_Prediction_v2/
├── scripts/
│   ├── 01_data_validation.py       # Data QC & manifest generation
│   ├── 02_kmer_extraction.py       # KMC-based k-mer counting
│   ├── 03_matrix_construction.py   # Sparse matrix construction
│   ├── 04_optimization.py          # Optuna HPO
│   └── 05_model_training.py        # Incremental XGBoost training
├── config/
│   ├── config.yaml                 # Global configuration
│   └── config_<antibiotic>.yaml    # Antibiotic-specific best params
├── data/                           # (gitignored) Genomic data & matrices
├── models/                         # (gitignored) Trained model artifacts
├── logs/                           # (gitignored) Run logs
├── requirements.txt
├── METHODOLOGY.md                  # Mathematical & biological deep-dive
└── README.md
```

---

## Key Design Decisions

| Challenge | Solution |
|-----------|----------|
| p ≫ n high-dimensional data | Sparse CSR matrices + chunked out-of-core loading |
| RAM constraints (8 GB) | `max_bin=2` (1-bit histograms) + chunked training |
| Imbalanced AMR classes | `scale_pos_weight` in XGBoost; threshold fixed at 0.5 |
| Overfitting in HPO | Early stopping determines `n_estimators`; Optuna does not search it |
| Catastrophic forgetting | Epoch-based training with shuffled chunk order per epoch |
| Skewed mini-batches | Stratified linspace chunk selection preserves resistance ratio |

---

## Mathematical Foundations

For a deep dive into the biological motivation, feature space mathematics, and statistical design decisions of this pipeline, see **[METHODOLOGY.md](METHODOLOGY.md)**.

---

## Citation

If you use this pipeline in your research, please cite:

```
@software{ML_AMR_Prediction_v2,
  author  = {Eren Demirbas},
  title   = {ML\_AMR\_Prediction\_v2: Out-of-Core XGBoost for AMR Prediction from WGS K-mers},
  year    = {2026},
  url     = {https://github.com/your-username/ML_AMR_Prediction_v2}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
