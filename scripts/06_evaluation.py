#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Model Performance Analysis Module

This script performs in-depth evaluation of the trained AMR prediction model,
including threshold optimization, detailed performance metrics, error analysis,
and clinical performance interpretation.

Configuration Architecture:
    This script uses a two-stage configuration loading system:
    1. Global Config (config.yaml): Project settings, paths, and target antibiotic
    2. Specific Config (config_{antibiotic}.yaml): Exact test set from stratified
       split performed during optimization (04_optimization.py)

Evaluation Framework:
    1. Model loading from antibiotic-specific directory
    2. Test data preparation using exact test files from config
    3. Threshold optimization using multiple metrics
    4. Comprehensive performance assessment
    5. Error analysis (false positives/negatives)
    6. Clinical interpretation of results

Clinical Context:
    In antimicrobial resistance prediction, we must balance:
    - Sensitivity: Catching resistant bacteria (false negatives costly)
    - Specificity: Avoiding unnecessary antibiotics (false positives wasteful)
    
    Threshold optimization helps find the right balance for clinical deployment.
"""

# ============================================================================
# LIBRARY IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
import xgboost as xgb
import yaml
from pathlib import Path
from scipy.sparse import load_npz, vstack
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    auc,
    cohen_kappa_score,
    balanced_accuracy_score,
    roc_curve,
    average_precision_score,
    precision_score,
    recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import sys


# ============================================================================
# LOAD CONFIGURATION FROM YAML
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

# Load configuration
if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Configuration file not found: {CONFIG_PATH}\n"
        f"Please ensure config.yaml exists in the config/ directory."
    )

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Extract configuration values
TARGET_ANTIBIOTIC = config['project']['target_antibiotic']
CHUNK_SIZE = config['preprocessing']['chunk_size']
N_JOBS = config['xgboost_params'].get('n_jobs', -1)

# Cross-platform paths (antibiotic-specific) - loaded from global config
MATRIX_DIR = PROJECT_ROOT / config['paths']['matrix_dir'].format(antibiotic=TARGET_ANTIBIOTIC)
MODELS_DIR = PROJECT_ROOT / config['paths']['models_dir'].format(antibiotic=TARGET_ANTIBIOTIC)
OUTPUT_DIR = PROJECT_ROOT / config['paths']['analysis_results_dir'].format(antibiotic=TARGET_ANTIBIOTIC)

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_y_chunk(y_all, chunk_id, chunk_size, total_len):
    """
    Extract label subset for a specific data chunk.
    
    Args:
        y_all: Complete array of all labels
        chunk_id: Chunk identifier (0-indexed)
        chunk_size: Number of samples per chunk
        total_len: Total number of samples
    
    Returns:
        Subset of labels for the specified chunk
    """
    start = chunk_id * chunk_size
    end = min((chunk_id + 1) * chunk_size, total_len)
    return y_all[start:end]


def load_test_files_from_config():
    """
    Load test file list from antibiotic-specific configuration.
    
    This function ensures evaluation uses the EXACT test set defined during
    optimization (04_optimization.py). This maintains:
    - Reproducibility: Same test set across all evaluation runs
    - Stratification: Representative test set (hard, medium, easy cases)
    - Isolation: Test set never seen during training or hyperparameter tuning
    
    Why This Matters:
        Using a different test set than optimization would:
        1. Break reproducibility (results vary by random seed)
        2. Violate data leakage prevention (if test overlaps with train)
        3. Lose stratification benefits (may over/under-represent difficulty)
    
    Returns:
        list: Test chunk filenames (strings) from stratified split
    
    Raises:
        FileNotFoundError: If antibiotic-specific config doesn't exist
        KeyError: If data_split section is missing
    """
    antibiotic_config_path = PROJECT_ROOT / "config" / f"config_{TARGET_ANTIBIOTIC}.yaml"
    
    if not antibiotic_config_path.exists():
        raise FileNotFoundError(
            f"Antibiotic-specific configuration not found: {antibiotic_config_path}\n\n"
            f"This file is generated by hyperparameter optimization.\n"
            f"Please run optimization for '{TARGET_ANTIBIOTIC}' first:\n\n"
            f"  python scripts/04_optimization.py\n\n"
            f"This will create: config/config_{TARGET_ANTIBIOTIC}.yaml\n"
            f"containing the stratified test set for evaluation."
        )
    
    try:
        with open(antibiotic_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if 'data_split' not in config:
            raise KeyError(
                f"'data_split' section not found in {antibiotic_config_path.name}\n"
                f"The config file may be from an old version. Re-run optimization:\n"
                f"  python scripts/04_optimization.py"
            )
        
        test_files = config['data_split'].get('test_files', [])
        
        # Load the unbiased threshold calculated from training prevalence
        optimal_threshold = float(config.get('evaluation', {}).get('optimal_threshold', 0.5))
        
        if not test_files:
            raise ValueError(
                "No test files found in data_split section.\n"
                "Re-run optimization: python scripts/04_optimization.py"
            )
        
        print(f"✓ Loaded test set from: {antibiotic_config_path.name}")
        print(f"  Split method: {config['data_split'].get('split_method', 'N/A')}")
        print(f"  Test chunks: {len(test_files)}")
        print(f"  Unbiased Threshold: {optimal_threshold:.4f}")
        
        return test_files, optimal_threshold
        
    except yaml.YAMLError as e:
        print(f"ERROR: Malformed YAML file: {antibiotic_config_path}")
        print(f"  Details: {e}")
        sys.exit(1)
    except (KeyError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def get_y_chunk_legacy(y_all, chunk_id, chunk_size, total_len):
    """
    Extract label subset for a specific data chunk.
    
    Args:
        y_all: Complete array of all labels
        chunk_id: Chunk identifier (0-indexed)
        chunk_size: Number of samples per chunk
        total_len: Total number of samples
    
    Returns:
        Subset of labels for the specified chunk
    """
    start = chunk_id * chunk_size
    end = min((chunk_id + 1) * chunk_size, total_len)
    return y_all[start:end]


def load_test_data(y_all, all_chunk_files, test_filenames):
    """
    Load and aggregate test data from config-specified chunk files.
    
    Uses the exact test set defined during optimization (stratified split)
    to ensure evaluation is performed on the representative sample that
    includes hard, medium, and easy resistance cases.
    
    CRITICAL: This function uses the EXACT file list from config with NO
    randomization or reordering. The test set is fixed during optimization.
    
    Args:
        y_all: Complete label array
        all_chunk_files: List of ALL available matrix chunk file paths
        test_filenames: List of test chunk filenames from config (exact order preserved)
    
    Returns:
        tuple: (X_test, y_test, genome_ids_test)
            - X_test: Stacked sparse feature matrix
            - y_test: Corresponding labels
            - genome_ids_test: Genome identifiers (for error tracing)
    """
    print("\n" + "=" * 80)
    print("LOADING TEST DATA (FROM STRATIFIED SPLIT)")
    print("=" * 80)
    
    # Create filename -> path mapping
    file_map = {f.name: f for f in all_chunk_files}
    
    # Resolve test files from config (EXACT order preserved)
    test_files = []
    for fname in test_filenames:
        if fname in file_map:
            test_files.append(file_map[fname])
        else:
            print(f"WARNING: Test file '{fname}' not found in matrix directory")
    
    if not test_files:
        raise FileNotFoundError(
            "No test files could be resolved from config.\n"
            "Check that matrix chunks exist and match the config file list."
        )
    
    print(f"Test chunks: {[f.name for f in test_files]}")
    print(f"Total test chunks: {len(test_files)}")
    
    X_test_list = []
    y_test_list = []
    genome_ids_test = []
    
    # Load genome IDs
    try:
        genomes_file = MATRIX_DIR / f"genomes_{TARGET_ANTIBIOTIC}.csv"
        if not genomes_file.exists():
            raise FileNotFoundError(f"Genome ID file not found: {genomes_file}")
        
        genomes_df = pd.read_csv(genomes_file, encoding='utf-8')
        all_genome_ids = genomes_df['Genome ID'].values
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load genome IDs: {e}")
        sys.exit(1)
    
    # Load each test chunk
    for chunk_file in test_files:
        try:
            X_chunk = load_npz(chunk_file)
            chunk_num = int(chunk_file.stem.split('_')[-1])
            
            # Get corresponding labels and genome IDs
            y_chunk = get_y_chunk(y_all, chunk_num, CHUNK_SIZE, len(y_all))
            genome_ids_chunk = get_y_chunk(all_genome_ids, chunk_num, CHUNK_SIZE, len(all_genome_ids))
            
            X_test_list.append(X_chunk)
            y_test_list.append(y_chunk)
            genome_ids_test.extend(genome_ids_chunk)
            
            print(f"  ✓ Loaded {chunk_file.name}: {X_chunk.shape[0]} samples")
            
        except Exception as e:
            print(f"ERROR: Failed to load {chunk_file.name}: {e}")
            sys.exit(1)
    
    # Stack all chunks
    X_test = vstack(X_test_list)
    y_test = np.concatenate(y_test_list)
    
    print(f"\n✓ Total test data: {X_test.shape}")
    print(f"  Resistant: {np.sum(y_test)} | Susceptible: {len(y_test) - np.sum(y_test)}")
    print("=" * 80)
    
    return X_test, y_test, np.array(genome_ids_test)



# ============================================================================
# Threshold logic is now handled in 05_model_training.py to prevent Data Leakage.
# We strictly use the Training Set prevalence threshold for unseen test data.
# ============================================================================



# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_confusion_matrix_enhanced(y_true, y_pred, output_dir, antibiotic):
    """
    Generate publication-quality confusion matrix heatmap.
    
    Args:
        y_true (ndarray): Ground truth labels
        y_pred (ndarray): Predicted labels
        output_dir (Path): Directory to save the plot
        antibiotic (str): Antibiotic name for title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar_kws={'label': 'Count'},
        square=True,
        linewidths=1,
        linecolor='gray'
    )
    
    plt.title(f'Confusion Matrix: {antibiotic.title()} Resistance Prediction', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks([0.5, 1.5], ['Susceptible', 'Resistant'], rotation=0)
    plt.yticks([0.5, 1.5], ['Susceptible', 'Resistant'], rotation=0)
    
    plt.tight_layout()
    output_path = output_dir / f'confusion_matrix_{antibiotic}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Confusion matrix saved: {output_path.name}")


def plot_roc_curve_analysis(y_true, y_prob, output_dir, antibiotic):
    """
    Generate ROC curve with AUC score.
    
    Args:
        y_true (ndarray): Ground truth labels
        y_prob (ndarray): Predicted probabilities
        output_dir (Path): Directory to save the plot
        antibiotic (str): Antibiotic name for title
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#2E86AB', linewidth=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Random Classifier')
    
    plt.fill_between(fpr, tpr, alpha=0.2, color='#2E86AB')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve: {antibiotic.title()} Resistance Prediction', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3, linestyle=':', linewidth=0.8)
    
    plt.tight_layout()
    output_path = output_dir / f'roc_curve_{antibiotic}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ ROC curve saved: {output_path.name}")


def plot_precision_recall_curve_analysis(y_true, y_prob, output_dir, antibiotic):
    """
    Generate Precision-Recall curve with Average Precision score.
    
    Args:
        y_true (ndarray): Ground truth labels
        y_prob (ndarray): Predicted probabilities
        output_dir (Path): Directory to save the plot
        antibiotic (str): Antibiotic name for title
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    # Calculate baseline (prevalence)
    baseline = np.sum(y_true) / len(y_true)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='#A23B72', linewidth=2.5, 
             label=f'PR Curve (AP = {avg_precision:.4f})')
    plt.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5, 
                label=f'Baseline (Prevalence = {baseline:.4f})')
    
    plt.fill_between(recall, precision, alpha=0.2, color='#A23B72')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve: {antibiotic.title()} Resistance', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=11)
    plt.grid(alpha=0.3, linestyle=':', linewidth=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.tight_layout()
    output_path = output_dir / f'precision_recall_curve_{antibiotic}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Precision-Recall curve saved: {output_path.name}")


def plot_threshold_analysis(results_df, output_dir, antibiotic):
    """
    Generate multi-line plot showing how metrics vary with threshold.
    
    Args:
        results_df (DataFrame): Threshold analysis results from find_best_threshold
        output_dir (Path): Directory to save the plot
        antibiotic (str): Antibiotic name for title
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(results_df['threshold'], results_df['f1'], 
             marker='o', linewidth=2, markersize=4, label='F1 Score', color='#06A77D')
    plt.plot(results_df['threshold'], results_df['mcc'], 
             marker='s', linewidth=2, markersize=4, label='MCC', color='#D62246')
    plt.plot(results_df['threshold'], results_df['recall'], 
             marker='^', linewidth=2, markersize=4, label='Recall', color='#4E8098')
    plt.plot(results_df['threshold'], results_df['precision'], 
             marker='d', linewidth=2, markersize=4, label='Precision', color='#F18F01')
    
    # Mark optimal threshold
    best_idx = results_df['mcc'].idxmax()
    best_thresh = results_df.loc[best_idx, 'threshold']
    best_mcc = results_df.loc[best_idx, 'mcc']
    
    plt.axvline(x=best_thresh, color='red', linestyle='--', linewidth=1.5, 
                label=f'Optimal Threshold = {best_thresh:.2f}')
    
    plt.xlabel('Classification Threshold', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title(f'Threshold Sensitivity Analysis: {antibiotic.title()} Resistance', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=10, ncol=2)
    plt.grid(alpha=0.3, linestyle=':', linewidth=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    
    plt.tight_layout()
    output_path = output_dir / f'threshold_analysis_{antibiotic}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Threshold analysis saved: {output_path.name}")


def save_comprehensive_metrics(y_true, y_pred, y_prob, best_thresh, output_dir, antibiotic):
    """
    Export comprehensive evaluation metrics to CSV.
    
    Args:
        y_true (ndarray): Ground truth labels
        y_pred (ndarray): Predicted labels (at optimal threshold)
        y_prob (ndarray): Predicted probabilities
        best_thresh (float): Optimal classification threshold
        output_dir (Path): Directory to save the CSV
        antibiotic (str): Antibiotic name
    """
    # Confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Compute all metrics
    metrics = {
        'Antibiotic': antibiotic.title(),
        'Optimal_Threshold': best_thresh,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Sensitivity_Recall_TPR': recall_score(y_true, y_pred),
        'Specificity_TNR': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'Precision_PPV': precision_score(y_true, y_pred),
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'F1_Score': f1_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred),
        'Cohen_Kappa': cohen_kappa_score(y_true, y_pred),
        'ROC_AUC': roc_auc_score(y_true, y_prob),
        'PR_AUC': average_precision_score(y_true, y_prob),
        'True_Positives': int(tp),
        'True_Negatives': int(tn),
        'False_Positives': int(fp),
        'False_Negatives': int(fn),
        'Total_Samples': len(y_true),
        'Resistant_Count': int(np.sum(y_true)),
        'Susceptible_Count': int(len(y_true) - np.sum(y_true))
    }
    
    metrics_df = pd.DataFrame([metrics])
    output_path = output_dir / f'comprehensive_metrics_{antibiotic}.csv'
    metrics_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n  ✓ Comprehensive metrics saved: {output_path.name}")
    print("\nPublication-Ready Metrics Summary:")
    print("=" * 80)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.4f}")
        else:
            print(f"{key:30s}: {value}")
    print("=" * 80)


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================
def main():
    """
    Execute comprehensive model performance analysis.
    
    Pipeline:
        1. Load test data and trained model
        2. Generate predictions
        3. Optimize classification threshold
        4. Compute comprehensive performance metrics
        5. Perform error analysis
        6. Export results
    """
    # Set publication-quality matplotlib/seaborn styling
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.2)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    print("=" * 80)
    print(f"COMPREHENSIVE MODEL ANALYSIS: {config['project']['model'].upper()}")
    print("=" * 80)
    
    # ------------------------------------------------------------------------
    # STEP 1: Validate and Load Files
    # ------------------------------------------------------------------------
    print("\n[STEP 1/6] Validating data files...")
    print("=" * 80)
    
    try:
        # Check label file
        y_path = MATRIX_DIR / f"y_{TARGET_ANTIBIOTIC}.csv"
        if not y_path.exists():
            raise FileNotFoundError(f"Label file not found: {y_path}")
        
        # Check model file - try possible model name variations
        model_paths_to_try = [
            MODELS_DIR / f"xgboost_{TARGET_ANTIBIOTIC}_final_v2.json",
            MODELS_DIR / f"xgboost_{TARGET_ANTIBIOTIC}.json",
            MODELS_DIR / f"xgboost_{TARGET_ANTIBIOTIC}_incremental.json"
        ]
        
        model_path = None
        for path in model_paths_to_try:
            if path.exists():
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(
                "Model file not found. Tried:\n" +
                "\n".join([f"  - {p}" for p in model_paths_to_try])
            )
        
        print(f"  ✓ Label file: {y_path.name}")
        print(f"  ✓ Model file: {model_path.name}")
        
        # Load labels
        y_all = pd.read_csv(y_path, encoding='utf-8')['label'].values
        
        # Find matrix chunks
        chunk_files = sorted(
            list(MATRIX_DIR.glob(f"X_{TARGET_ANTIBIOTIC}_part_*.npz")),
            key=lambda x: int(x.stem.split('_')[-1])
        )
        
        if not chunk_files:
            raise FileNotFoundError(
                f"No matrix chunks found matching: X_{TARGET_ANTIBIOTIC}_part_*.npz"
            )
        
        print(f"  ✓ Matrix chunks: {len(chunk_files)} files")
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\\nPlease ensure you have run:")
        print("  1. Matrix creation: 03_matrix_construction.py")
        print("  2. Model training: 05_model_training.py")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        sys.exit(1)
    
    # ------------------------------------------------------------------------
    # STEP 2: Load Data and Model
    # ------------------------------------------------------------------------
    print("\n[STEP 2/6] Loading data and model...")

    # Load test filenames and threshold from config
    test_filenames, best_thresh = load_test_files_from_config()

    # Load test data
    X_test, y_test, ids_test = load_test_data(y_all, chunk_files, test_filenames)

    # Load model
    try:
        print(f"\nLoading model: {model_path.name}")
        model = xgb.XGBClassifier(n_jobs=N_JOBS)
        model.load_model(str(model_path))
        print("  ✓ Model loaded successfully")
        print(f"  ✓ Number of features: {model.n_features_in_}")
        print(f"  ✓ Number of trees: {model.n_estimators}")
        
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)
    
    # ------------------------------------------------------------------------
    # STEP 4: Generate Predictions
    # ------------------------------------------------------------------------
    print("\n[STEP 4/6] Generating predictions...")
    
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        print(f"  ✓ Predictions generated for {len(y_prob)} samples")
        print(f"  ✓ Probability range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")
        
    except Exception as e:
        print(f"ERROR: Prediction failed: {e}")
        sys.exit(1)
    
    # ------------------------------------------------------------------------
    # STEP 5: Applying Unbiased Threshold and Performance Evaluation
    # ------------------------------------------------------------------------
    print("\n[STEP 5/6] Evaluating performance with unbiased training threshold...")
    print(f"  Applying Threshold: {best_thresh:.4f}")
    
    y_pred_opt = (y_prob >= best_thresh).astype(int)
    
    # Compute comprehensive metrics
    print("\\n" + "=" * 80)
    print("FINAL PERFORMANCE METRICS")
    print("=" * 80)
    
    acc = accuracy_score(y_test, y_pred_opt)
    balanced_acc = balanced_accuracy_score(y_test, y_pred_opt)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    
    kappa = cohen_kappa_score(y_test, y_pred_opt)
    mcc = matthews_corrcoef(y_test, y_pred_opt)
    
    print(f"Accuracy                   : {acc:.4f}")
    print(f"Balanced Accuracy          : {balanced_acc:.4f}")
    print(f"ROC AUC Score              : {roc_auc:.4f}")
    print(f"Precision-Recall AUC       : {pr_auc:.4f}")
    print(f"Cohen's Kappa              : {kappa:.4f}")
    print(f"Matthews Correlation Coef  : {mcc:.4f}")
    print(f"Optimal Threshold          : {best_thresh:.2f}")
    print("=" * 80)
    
    print("\\nDetailed Classification Report:")
    print("-" * 80)
    print(classification_report(
        y_test,
        y_pred_opt,
        target_names=['Susceptible (0)', 'Resistant (1)'],
        digits=4
    ))
    
    print("Confusion Matrix:")
    print("-" * 80)
    cm = confusion_matrix(y_test, y_pred_opt)
    print("                     Predicted")
    print("                   Susceptible  Resistant")
    print(f"Actual Susceptible     {cm[0,0]:<8}     {cm[0,1]:<8}")
    print(f"       Resistant       {cm[1,0]:<8}     {cm[1,1]:<8}")
    print("-" * 80)
    
    # Clinical interpretation
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    print("\\nClinical Performance Metrics:")
    print("-" * 80)
    print(f"Sensitivity (True Positive Rate):  {sensitivity:.4f}")
    print("  → Proportion of resistant bacteria correctly identified")
    print(f"Specificity (True Negative Rate):  {specificity:.4f}")
    print("  → Proportion of susceptible bacteria correctly identified")
    print(f"Positive Predictive Value (PPV):   {ppv:.4f}")
    print("  → When model predicts resistant, probability it's correct")
    print(f"Negative Predictive Value (NPV):   {npv:.4f}")
    print("  → When model predicts susceptible, probability it's correct")
    print("-" * 80)
    
    # ------------------------------------------------------------------------
    # STEP 6: Generate Publication-Quality Visualizations
    # ------------------------------------------------------------------------
    print("\n[STEP 6/6] Generating publication-quality outputs...")
    print("=" * 80)
    
    # Save comprehensive metrics CSV
    save_comprehensive_metrics(
        y_test, 
        y_pred_opt, 
        y_prob, 
        best_thresh, 
        OUTPUT_DIR, 
        TARGET_ANTIBIOTIC
    )
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix_enhanced(y_test, y_pred_opt, OUTPUT_DIR, TARGET_ANTIBIOTIC)
    plot_roc_curve_analysis(y_test, y_prob, OUTPUT_DIR, TARGET_ANTIBIOTIC)
    plot_precision_recall_curve_analysis(y_test, y_prob, OUTPUT_DIR, TARGET_ANTIBIOTIC)
    
    print("\n" + "=" * 80)
    print("ALL PUBLICATION-READY OUTPUTS GENERATED")
    print("=" * 80)
    
    # ------------------------------------------------------------------------
    # STEP 6: Error Analysis
    # ------------------------------------------------------------------------
    print("\\n[STEP 6/6] Performing error analysis...")
    print("=" * 80)
    
    # False Negatives (most critical in medical context)
    fn_mask = (y_test == 1) & (y_pred_opt == 0)
    fn_ids = ids_test[fn_mask]
    fn_probs = y_prob[fn_mask]
    
    print(f"False Negatives (Missed Resistant): {len(fn_ids)}")
    
    if len(fn_ids) > 0:
        error_df = pd.DataFrame({
            'Genome_ID': fn_ids,
            'Predicted_Probability': fn_probs,
            'True_Label': 'Resistant',
            'Predicted_Label': 'Susceptible',
            'Error_Type': 'False_Negative'
        })
        
        print("\\nTop 10 False Negatives (by probability):")
        print(error_df.sort_values('Predicted_Probability').head(10).to_string(index=False))
        
        error_file = OUTPUT_DIR / "error_analysis_false_negatives.csv"
        error_df.to_csv(error_file, index=False, encoding='utf-8')
        print(f"\\n  ✓ Full error analysis saved: {error_file}")
    
    # False Positives
    fp_mask = (y_test == 0) & (y_pred_opt == 1)
    fp_count = np.sum(fp_mask)
    print(f"\\nFalse Positives (Overestimated): {fp_count}")
    
    print("\\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\\nResults saved to:")
    print(f"  {OUTPUT_DIR}")
    print("\\nNext steps:")
    print("  1. Review false negatives - these are high-risk predictions")
    print("  2. Consider adjusting threshold if clinical context demands higher sensitivity")
    print("  3. Run feature extraction: 07_explainability.py")
    print("=" * 80)


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()
