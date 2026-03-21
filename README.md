# ML AMR Prediction Framework: Alignment-Free WGS Out-of-Core Learning

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost-orange.svg)
![Pipeline](https://img.shields.io/badge/pipeline-Nextflow-brightgreen.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Abstract
The **ML AMR Prediction Framework** is a highly scalable, multi-antibiotic machine learning pipeline designed to predict Antimicrobial Resistance (AMR) directly from raw Whole Genome Sequencing (WGS) data. Utilizing a rapid, alignment-free k-mer extraction methodology, the framework discovers the underlying biological resistance mechanisms autonomously, entirely bypassing reference genomes. Its highly extensible architecture enables predicting resistance across ANY antibiotic and pathogen pair with optimal performance. Currently, the framework utilizes **Ciprofloxacin** as the primary capability benchmark to demonstrate the power of the end-to-end analytical pipeline.

---

## The Three Core Pillars

### 1. The Computer Science / Informatics
Dealing with high-dimensional genomic features often results in severe computational bottlenecks. This framework addresses the RAM bottleneck by orchestrating hardware-efficient strategies:
- **Fast K-mer Extraction:** We leverage `KMC` (K-mer Counter) for highly compressed, distributed counting of 21-mers, allowing for linear scaling and minimal memory overhead.
- **Out-of-Core Learning:** By implementing dynamic stratified chunking and storing intermediate data in highly compressed `.npz` structures, the pipeline effortlessly handles datasets that exceed available RAM capacity.
- **Memory-Efficient Processing:** Model training utilizes iterative data loading via `XGBoost DMatrix`, supporting training over matrices of **48+ million features** seamlessly on standard consumer hardware (e.g., standard M4 Pro architecture).

### 2. The Mathematics
To maximize predictive reliability in a clinical context, the optimization algorithms have been heavily fine-tuned:
- **Bayesian Hyperparameter Optimization:** `Optuna` is integrated natively to traverse the complex loss landscape, finding optimal parameters faster while mitigating the risk of overfitting.
- **Dynamic Classification Thresholding:** Recognizing the risks of AMR, the threshold dynamically shifts to explicitly prioritize **Clinical Sensitivity** prioritizing the minimization of False Negatives (FNs).
- **Embedded Feature Selection:** Exploring importance via the internal **Gain** metric, the algorithms extract structural genomic features mathematically proven to govern resistance phenotypes.

### 3. The Biology
The ultimate goal of the framework is automated biological mechanism discovery and robust Genotype-Phenotype mapping:
- **Reverse Translating Models to Biology:** Extracted high-impact mathematical features (the top 21-mers) are translated directly into biological significance.
- **Automated Nextflow Discovery:** Using `08_blast_pipeline.nf`, we automate queries against the local **CARD** database and the remote **NCBI `nt`** repository.
- **Reference-Free Discovery:** The framework has successfully and autonomously rediscovered established resistance mechanisms entirely from scratch without aligning to any existing reference. Notable rediscoveries include *gyrA* QRDR mutations, *OXA-909* plasmids, and *msbA* efflux pumps. 

---

## Project Architecture
The project strictly complies with the **Cookiecutter Data Science** standard to enforce reproducibility and structural consistency across the repository. 

Key Data Streams:
- `data/raw/`: Immutable raw genomes (`.fna`).
- `data/external/`: Metadata and third-party BLAST databases.
- `data/interim/`: Intermediate representations, global KMC indices.
- `data/processed/`: Final, canonical matrices prepared for model injection.
- `results/`: Centralised analysis outputs and visualizations (replaces `analysis_results/`).

**Dynamic Multiprocessing via Config:**
The framework utilizes `config/config.yaml` to govern variable generation. By isolating file path structures dynamically based on the target `{antibiotic}` flag, users can launch fully parallel, multi-antibiotic runs without data collision.

---

## Pipeline Workflow

The complete end-to-end execution is governed by a sequence of highly modular analytical scripts:

| Step | Script | Description |
| :--- | :--- | :--- |
| **01** | `01_data_validation.py` / `01b_data_validation.py` | Validates initial metadata consistency and prepares genome manifests. |
| **02** | `02_kmer_extraction.py` / `02b_global_qc_analysis.py` | Employs KMC to extract raw k-mers and plot global distribution metrics. |
| **03** | `03_matrix_construction.py` / `03b_matrix_validation_qc.py` | Transforms strings into sparse frequency matrices and flags QC outliers. |
| **04** | `04_optimization.py` | Performs `Optuna` Bayesian Hyperparameter Optimization. |
| **05** | `05_model_training.py` | Invokes the Out-of-Core XGBoost trainer on the generated `.npz` chunking. |
| **06** | `06_evaluation.py` | Evaluates accuracy, clinical sensitivity, ROC/PR curves, and dynamic limits. |
| **07** | `07_explainability.py` | Leverages model **Gain** to unpack the highest impact genomic 21-mers. |
| **08** | `08_blast_annotation.py` / `08_blast_pipeline.nf` | Annotates highest impact variables against the local CARD & NCBI databases. |

---

## Case Study: Ciprofloxacin Validation
Currently run against a standard benchmark of Ciprofloxacin resistance, our zero-alignment, feature-extracted methodology successfully yields state-of-the-art predictive performance:
- **Accuracy:** 96.8%
- **Clinical Sensitivity:** 97.2%
- **ROC-AUC:** 0.99

This establishes a profound proof-of-concept validation of the Out-of-Core framework logic to scale toward other targets rapidly.

---

## Installation & Quickstart

### Dependencies
Before spinning up the framework locally, ensure you have the core suite of command line dependencies natively installed.
- **Python:** `3.9+`
- **KMC (K-mer Counter):** Used for genomic k-mer mining.
- **BLAST+:** Mandatory for sequence homology queries.
- **Nextflow:** Required to launch the `08_blast_pipeline.nf` workflow.
- **Hardware Profile:** Tested successfully on an Apple M4 Pro (24GB RAM). The out-of-core DMatrix chunking mechanism allows processing of 48M+ features on standard local workstations without requiring high-performance computing (HPC) clusters.

### Quickstart Execution
1. **Clone & Standardize:**
   ```bash
   git clone <repository_url>
   cd ML_AMR_Prediction_v2
   pip install -r requirements.txt
   ```
2. **Execute sequential processing** (Modify `config.yaml` to specify the target antibiotic path as required):
   ```bash
   python scripts/01_data_validation.py
   python scripts/02_kmer_extraction.py
   python scripts/03_matrix_construction.py
   python scripts/04_optimization.py
   python scripts/05_model_training.py
   python scripts/06_evaluation.py
   python scripts/07_explainability.py
   python scripts/08_blast_annotation.py
   ```
