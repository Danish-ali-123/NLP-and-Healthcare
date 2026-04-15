#!/usr/bin/env python3

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_score, recall_score
)
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import MetricCalculator if available
try:
    from src.models.metrics import MetricCalculator
    HAS_METRIC_CALCULATOR = True
except ImportError:
    HAS_METRIC_CALCULATOR = False
    print("Warning: MetricCalculator not found. Using custom ECE implementation.")

# Define directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Define the 6-class label mapping based on the data
def get_label_mapping():
    """Return the label mapping for 6-class classification (with multilingual support)."""
    return {
        "Bone-related disorders": 0,
        "Hip-related disorders": 1,
        "Musculoskeletal disorders": 2,
        "Other": 3,
        "Spinal disorders": 4,
        "Unknown": 5,
        # Hindi labels
        "हड्डी संबंधित विकार": 0,
        "कूल्हे से संबंधित विकार": 1,
        "मस्कुलोस्केलेटल विकार": 2,
        "अन्य": 3,
        "रीढ़ से संबंधित विकार": 4,
        "अगਿਆत": 5,
        # Punjabi labels
        "ਹੱਡੀਆਂ ਨਾਲ ਸਬੰਧਤ ਵਿਕਾਰ": 0,
        "ਕੂਲ੍ਹੇ ਨਾਲ ਸਬੰਧਤ ਵਿਕਾਰ": 1,
        "ਮਸੂਕਲੋਸਕੇਲਟਲ ਵਿਕਾਰ": 2,
        "ਹੋਰ": 3,
        "ਕਮਰ ਨਾਲ ਸਬੰਧਤ ਵਿਕਾਰ": 4,
    }

def load_predictions(file_path: str) -> pd.DataFrame:
    """Load predictions from JSONL file into a pandas DataFrame."""
    predictions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return pd.DataFrame(predictions)

def custom_ece(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE) if MetricCalculator is not available."""
    # Create bins of confidence scores
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba.max(axis=1), bins, right=True)
    
    ece = 0.0
    for i in range(1, n_bins + 1):
        mask = (bin_indices == i)
        if not np.any(mask):
            continue
        
        # Calculate mean confidence and accuracy for this bin
        conf = y_pred_proba[mask].max(axis=1).mean()
        acc = (y_true[mask] == y_pred_proba[mask].argmax(axis=1)).mean()
        
        # Weight by the number of samples in the bin
        weight = mask.sum() / len(y_true)
        ece += weight * abs(conf - acc)
    
    return ece

def compute_metrics_for_model(predictions_df: pd.DataFrame, label_mapping: Dict[str, int]) -> Dict[str, float]:
    """Compute all required metrics for a model's predictions."""
    # Map labels to integers
    predictions_df['true_label_id'] = predictions_df['true_label'].map(label_mapping)
    predictions_df['pred_label_id'] = predictions_df['label'].map(label_mapping)
    
    # Handle unknown labels by mapping to 'Other' (id 3)
    predictions_df['true_label_id'] = predictions_df['true_label_id'].fillna(3)
    predictions_df['pred_label_id'] = predictions_df['pred_label_id'].fillna(3)
    
    # Convert to numpy arrays
    y_true = predictions_df['true_label_id'].astype(int).values
    y_pred = predictions_df['pred_label_id'].astype(int).values
    
    # Create one-hot encoded labels for AUROC and AUPRC
    n_classes = len(label_mapping)
    y_true_onehot = np.eye(n_classes)[y_true]
    
    # Create probability matrix from confidence scores
    # We'll use the confidence score for the predicted class and distribute remaining probability equally
    y_pred_proba = np.zeros((len(predictions_df), n_classes))
    for idx, (i, row) in enumerate(predictions_df.iterrows()):
        pred_id = row['pred_label_id']
        conf = row['confidence']
        remaining = (1.0 - conf) / (n_classes - 1)
        y_pred_proba[idx] = remaining
        y_pred_proba[idx, int(pred_id)] = conf
    
    # Compute metrics
    metrics = {
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
    }
    
    # Compute AUROC (one-vs-rest)
    try:
        auroc = roc_auc_score(y_true_onehot, y_pred_proba, average='macro', multi_class='ovr')
        metrics['auroc_macro'] = auroc
    except ValueError as e:
        print(f"AUROC calculation failed: {e}")
        metrics['auroc_macro'] = 0.0
    
    # Compute AUPRC
    try:
        auprc = average_precision_score(y_true_onehot, y_pred_proba, average='macro')
        metrics['auprc_macro'] = auprc
    except ValueError as e:
        print(f"AUPRC calculation failed: {e}")
        metrics['auprc_macro'] = 0.0
    
    # Compute ECE
    if HAS_METRIC_CALCULATOR:
        try:
            calculator = MetricCalculator()
            ece = calculator.calculate_ece(y_pred_proba, y_true)
            metrics['ece'] = ece
        except Exception as e:
            print(f"ECE calculation using MetricCalculator failed: {e}")
            metrics['ece'] = custom_ece(y_true, y_pred_proba)
    else:
        metrics['ece'] = custom_ece(y_true, y_pred_proba)
    
    return metrics

def evaluate_all_models() -> pd.DataFrame:
    """Evaluate all models in the results directory."""
    label_mapping = get_label_mapping()
    all_metrics = []
    
    # Get all JSONL files in the results directory
    prediction_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('_predictions.jsonl')]
    
    for file in prediction_files:
        model_name = file.replace('_predictions.jsonl', '')
        file_path = os.path.join(RESULTS_DIR, file)
        
        print(f"\n=== Evaluating {model_name} ===")
        
        # Load predictions
        predictions_df = load_predictions(file_path)
        print(f"Loaded {len(predictions_df)} samples")
        
        # Compute overall metrics
        overall_metrics = compute_metrics_for_model(predictions_df, label_mapping)
        overall_metrics['model'] = model_name
        overall_metrics['language'] = 'all'
        all_metrics.append(overall_metrics)
        
        # Print overall metrics
        print("Overall Metrics:")
        for metric, value in overall_metrics.items():
            if metric not in ['model', 'language']:
                print(f"  {metric}: {value:.4f}")
        
        # Compute metrics per language
        languages = predictions_df['language'].unique()
        for lang in languages:
            lang_df = predictions_df[predictions_df['language'] == lang]
            lang_metrics = compute_metrics_for_model(lang_df, label_mapping)
            lang_metrics['model'] = model_name
            lang_metrics['language'] = lang
            all_metrics.append(lang_metrics)
            
            # Print language-specific metrics
            print(f"\n{lang} Metrics:")
            for metric, value in lang_metrics.items():
                if metric not in ['model', 'language']:
                    print(f"  {metric}: {value:.4f}")
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save metrics to CSV
    metrics_csv = os.path.join(RESULTS_DIR, 'llm_performance_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\nMetrics saved to: {metrics_csv}")
    
    return metrics_df

def generate_summary_table(llm_metrics_df: pd.DataFrame):
    """Generate a summary table aligned with existing model results."""
    # Load existing model results for alignment
    existing_results_file = os.path.join(BASE_DIR, "..", "results_jan_26", "model_performance_detailed.csv")
    if os.path.exists(existing_results_file):
        existing_df = pd.read_csv(existing_results_file)
        print(f"Loaded existing results: {len(existing_df)} rows")
        
        # Combine with LLM results
        combined_df = pd.concat([existing_df, llm_metrics_df], ignore_index=True)
    else:
        print("Existing results file not found. Using only LLM results.")
        combined_df = llm_metrics_df.copy()
    
    # Reorder columns to match existing format
    desired_columns = ['model', 'language', 'f1_macro', 'auroc_macro', 'auprc_macro', 'ece', 'accuracy']
    # Ensure all columns are present
    for col in desired_columns:
        if col not in combined_df.columns:
            combined_df[col] = 0.0
    
    # Filter to desired columns
    combined_df = combined_df[desired_columns]
    
    # Save combined results
    summary_file = os.path.join(RESULTS_DIR, 'combined_performance_summary.csv')
    combined_df.to_csv(summary_file, index=False)
    print(f"Combined summary saved to: {summary_file}")
    
    # Generate a markdown report
    generate_markdown_report(combined_df)

def generate_markdown_report(metrics_df: pd.DataFrame):
    """Generate a markdown report with performance summary."""
    report_file = os.path.join(RESULTS_DIR, 'llm_evaluation_report.md')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Open Source LLM Evaluation Report\n\n")
        f.write("## Zero-shot Clinical Decision System Performance\n\n")
        f.write("This report presents the performance of open-source LLMs as zero-shot clinical decision systems for orthopedic diagnosis.\n\n")
        
        # Overall performance table
        f.write("### Overall Performance (All Languages)\n\n")
        overall_df = metrics_df[metrics_df['language'] == 'all']
        f.write(overall_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")
        
        # Performance by language
        f.write("### Performance by Language\n\n")
        for lang in ['en', 'hi', 'pa']:
            lang_df = metrics_df[metrics_df['language'] == lang]
            f.write(f"#### {lang.upper()}\n\n")
            f.write(lang_df.to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")
        
        # Model comparison summary
        f.write("### Model Comparison Summary\n\n")
        f.write("| Metric | Best Model | Score | Second Best | Score | Third Best | Score |\n")
        f.write("|--------|------------|-------|-------------|-------|-------------|-------|\n")
        
        # Find best models for each metric
        metrics_to_compare = ['f1_macro', 'accuracy', 'auroc_macro', 'auprc_macro', 'ece']
        overall_df = metrics_df[metrics_df['language'] == 'all']
        
        for metric in metrics_to_compare:
            if metric in overall_df.columns:
                sorted_df = overall_df.sort_values(by=metric, ascending=(metric == 'ece'))
                if len(sorted_df) >= 3:
                    best = sorted_df.iloc[0]
                    second = sorted_df.iloc[1]
                    third = sorted_df.iloc[2]
                    f.write(f"| {metric} | {best['model']} | {best[metric]:.4f} | {second['model']} | {second[metric]:.4f} | {third['model']} | {third[metric]:.4f} |\n")
                elif len(sorted_df) == 2:
                    best = sorted_df.iloc[0]
                    second = sorted_df.iloc[1]
                    f.write(f"| {metric} | {best['model']} | {best[metric]:.4f} | {second['model']} | {second[metric]:.4f} | N/A | N/A |\n")
                elif len(sorted_df) == 1:
                    best = sorted_df.iloc[0]
                    f.write(f"| {metric} | {best['model']} | {best[metric]:.4f} | N/A | N/A | N/A | N/A |\n")
        
        f.write("\n")
        f.write("### Methodology\n\n")
        f.write("- **Models**: Mistral-7B-Instruct, LLaMA-3-8B-Instruct, Gemma-7B-Instruct, Zephyr-7B\n")
        f.write("- **Task**: Zero-shot clinical diagnosis\n")
        f.write("- **Input**: Clinical orthopedic text (English, Hindi, Punjabi)\n")
        f.write("- **Output**: Structured JSON with diagnosis and confidence score\n")
        f.write("- **Metrics**: F1-macro, Accuracy, Balanced Accuracy, AUROC, AUPRC, ECE\n")
        f.write("- **Test Data**: Same test split used for fine-tuned transformer models\n")
    
    print(f"Markdown report saved to: {report_file}")

def main():
    """Main function to run the evaluation."""
    print("=== Open Source LLM Evaluation Metrics Calculation ===")
    print(f"Results directory: {RESULTS_DIR}")
    
    # Evaluate all models
    llm_metrics_df = evaluate_all_models()
    
    # Generate summary table aligned with existing results
    generate_summary_table(llm_metrics_df)
    
    print("\n=== Evaluation Complete ===")

if __name__ == "__main__":
    main()
