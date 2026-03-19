# Quick Start Guide - AMR Prediction Pipeline

## Prerequisites Checklist

Before running the pipeline, ensure you have:

- [ ] Python 3.8 or higher installed
- [ ] All Python packages installed (`pip install -r requirements.txt`)
- [ ] KMC tool installed in `bin/` directory
- [ ] Genome assembly files (.fna) in `data/raw_genomes/`
- [ ] Metadata file in `data/metadata/genome_amr_matrix.csv`

## Step-by-Step Execution Guide

### Step 1: Validate Your Data

```bash
cd /path/to/ML_project
python scripts/00_check_data.py
```

**Expected output:**
- Total genome count
- Usable genomes (with both file and metadata)
- Resistance distribution for each antibiotic
- Recommendation for ciprofloxacin analysis

**Check for:**
- No "CRITICAL ERROR" messages
- At least 500+ usable genomes for ciprofloxacin
- Resistance ratio between 30-70% (ideal balance)

**If errors occur:**
- Verify genome files are named correctly (matching Genome ID in metadata)
- Check that metadata CSV has required columns

---

### Step 2: Count K-mers

```bash
python scripts/01_count_kmers.py
```

**What happens:**
- Processes each genome file individually
- Creates k-mer count databases (.kmc_pre and .kmc_suf files)
- Shows progress bar
- Can resume if interrupted

**Expected duration:** 1-5 minutes per genome (varies by size)

**Output location:** `data/kmc_outputs/`

**Check for:**
- "K-MER COUNTING OPERATION COMPLETE" message
- Failed count should be 0 or very low
- Files `<genome_id>.kmc_pre` and `<genome_id>.kmc_suf` created

**If errors occur:**
- Check KMC binary path in script
- Verify sufficient disk space
- Check genome file format (should be valid FASTA)

---

### Step 3: Create Feature Matrix

```bash
python scripts/02_create_matrix.py
```

**What happens:**
- Creates global k-mer vocabulary
- Builds sparse binary matrices in chunks
- Saves labels and genome IDs separately

**Expected duration:** 30 minutes to 2 hours (depending on data size)

**Output location:** `data/matrix/`

**Files created:**
- `features.txt` - K-mer dictionary
- `X_cipro_part_0.npz`, `X_cipro_part_1.npz`, etc. - Matrix chunks
- `y_cipro.csv` - Resistance labels
- `genomes_cipro.csv` - Genome identifiers

**Check for:**
- "FEATURE MATRIX CONSTRUCTION COMPLETE" message
- Multiple .npz chunk files created
- Feature count (should be hundreds of thousands to millions)
- Sparsity percentage (should be >95%)

**Memory note:** This step uses the most RAM. If you get memory errors, reduce CHUNK_SIZE in the script.

---

### Step 4: Train the Model

```bash
python scripts/03_train_final_direct.py
```

**What happens:**
- Shuffles and splits data into train/test sets
- Trains XGBoost model incrementally across chunks
- Evaluates on test set
- Optimizes classification threshold
- Saves trained model

**Expected duration:** 15-45 minutes

**Output location:** `models/xgboost_ciprofloxacin_final_v2.json`

**Check for:**
- "TRAINING COMPLETE" message
- Model performance metrics:
  - Accuracy > 0.90 (90% correct)
  - ROC AUC > 0.85 (excellent discrimination)
  - MCC > 0.70 (strong correlation)
- Model file saved successfully

**Expected performance:**
- Accuracy: 0.92-0.96
- ROC AUC: 0.90-0.98
- MCC: 0.75-0.90

**If performance is poor (<80% accuracy):**
- Check data quality (Step 1)
- Verify sufficient training samples (>500)
- Check class balance (30-70% resistant)
- Consider adjusting hyperparameters in config.yaml

---

### Step 5: Detailed Performance Analysis

```bash
python scripts/04_detailed_analysis.py
```

**What happens:**
- Loads model and test data
- Optimizes classification threshold
- Computes comprehensive metrics
- Identifies false negatives (missed resistant bacteria)
- Exports error analysis

**Expected duration:** 5-15 minutes

**Output location:** `analysis_results/error_analysis_false_negatives.csv`

**Check for:**
- Comprehensive metrics table
- Clinical performance interpretation
- Error analysis section
- False negative count (should be minimized)

**Key metrics to review:**
- Sensitivity: How many resistant bacteria are caught?
- Specificity: How many susceptible bacteria are correctly identified?
- PPV/NPV: Prediction reliability

---

### Step 6: Extract Important Features

```bash
python scripts/05_extract_top_features.py
```

**What happens:**
- Extracts feature importance from model
- Maps features to k-mer sequences
- Exports in CSV and FASTA formats

**Expected duration:** <5 minutes

**Output location:** `analysis_results/`

**Files created:**
- `top_50_features_cipro_final.csv` - Feature rankings
- `top_50_features_cipro_final.fasta` - Sequences for BLAST

**Next step:** Upload FASTA file to NCBI BLAST to identify genes

---

## Validation Checklist

After completing all steps, verify:

- [ ] All scripts completed without critical errors
- [ ] Model performance meets expectations (>90% accuracy)
- [ ] Feature importance files generated
- [ ] Error analysis shows acceptable false negative rate
- [ ] Output files are readable and contain expected data

## Common Issues and Solutions

### Issue: "No k-mer databases found"
**Solution:** Ensure Step 2 (count_kmers) completed successfully

### Issue: "Memory error" during matrix creation
**Solution:** Reduce CHUNK_SIZE in `02_create_matrix.py` (try 100 or 50)

### Issue: Model file not found in analysis script
**Solution:** Check that Step 4 (training) completed and saved model

### Issue: Low model performance
**Causes:**
1. Insufficient training data (<500 samples)
2. Extreme class imbalance (>90% or <10% resistant)
3. Low-quality genomes (contamination, poor assembly)
4. Incorrect metadata labels

**Solutions:**
1. Collect more data
2. Use different antibiotic with better balance
3. Filter low-quality genomes
4. Verify metadata accuracy

### Issue: Script crashes during training
**Solution:** 
1. Check available RAM (need 16GB+)
2. Close other applications
3. Reduce chunk size or test chunks

## Getting Help

If you encounter issues not covered here:

1. Check the full README.md for detailed documentation
2. Review script comments for specific function documentation
3. Verify all prerequisites are met
4. Check data file formats match expected structure

## Tips for Best Results

1. **Data Quality:** Clean, high-quality genome assemblies produce better models
2. **Class Balance:** 30-70% resistance ratio is ideal
3. **Sample Size:** More samples (>1000) improve generalization
4. **Feature Selection:** Minimum support (10 genomes) filters noise
5. **Validation:** Always review false negatives for clinical implications

---

**Remember:** This is a machine learning model for research purposes. Clinical decisions should involve multiple factors and expert interpretation.

## Next Steps After Pipeline Completion

1. **Biological Validation:**
   - BLAST top features against resistance gene databases
   - Literature review for identified genes
   - Compare with known resistance mechanisms

2. **Model Deployment:**
   - Test on independent dataset
   - Compare with existing methods
   - Document performance on different bacterial populations

3. **Further Analysis:**
   - Generate ROC curves and PR curves
   - Perform cross-validation
   - Test on other antibiotics
   - Compare different ML algorithms

---

**Document Version:** 1.0  
**Last Updated:** January 2025
