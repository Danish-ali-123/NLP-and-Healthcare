import json
import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

# Define the available diagnosis categories
AVAILABLE_LABELS = [
    "Bone-related disorders",
    "Hip-related disorders",
    "Musculoskeletal disorders",
    "Other",
    "Spinal disorders"
]

# Define the models to evaluate
MODEL_NAMES = {
    "deepseek-open": "DeepSeek Open",
    "mistral-7b-instruct": "Mistral 7B Instruct",
    "zephyr-7b": "Zephyr 7B"
}

def load_predictions(file_path):
    """Load predictions from a JSONL file."""
    predictions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return predictions

def compute_model_metrics(predictions):
    """Compute metrics for a single model's predictions."""
    y_true = []
    y_pred = []
    
    for pred in predictions:
        y_true.append(pred["true_label"])
        y_pred.append(pred["label"])
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate precision, recall, f1-score with macro averaging
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, 
        average='macro', 
        labels=AVAILABLE_LABELS,
        zero_division=0
    )
    
    # Calculate weighted f1-score
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, 
        average='weighted', 
        labels=AVAILABLE_LABELS,
        zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_weighted": f1_weighted,
        "y_true": y_true,
        "y_pred": y_pred
    }

def generate_classification_report(y_true, y_pred, model_name):
    """Generate a classification report for a model."""
    report = classification_report(
        y_true, y_pred,
        target_names=AVAILABLE_LABELS,
        labels=AVAILABLE_LABELS,
        zero_division=0
    )
    
    print(f"\n=== {model_name} Classification Report ===")
    print(report)
    
    return report

def main():
    """Main function to compute metrics for all models."""
    results_dir = "results"
    metrics = {}
    
    print("=== LLM Model Evaluation Metrics ===")
    print("Computing metrics for DeepSeek, Mistral 7B, and Zephyr 7B models...\n")
    
    # Process each model
    for model_key, model_name in MODEL_NAMES.items():
        file_path = os.path.join(results_dir, f"{model_key}_predictions.jsonl")
        
        if os.path.exists(file_path):
            print(f"Processing {model_name}...")
            predictions = load_predictions(file_path)
            model_metrics = compute_model_metrics(predictions)
            metrics[model_name] = model_metrics
            
            # Generate classification report
            generate_classification_report(
                model_metrics["y_true"], 
                model_metrics["y_pred"],
                model_name
            )
        else:
            print(f"⚠️  {model_name} predictions file not found: {file_path}")
    
    # Create a comparison table
    print("\n" + "="*70)
    print("Model Comparison - Macro Metrics")
    print("="*70)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-"*70)
    
    for model_name, model_metrics in metrics.items():
        print(f"{model_name:<25} "
              f"{model_metrics['accuracy']:<10.4f} "
              f"{model_metrics['precision']:<10.4f} "
              f"{model_metrics['recall']:<10.4f} "
              f"{model_metrics['f1']:<10.4f}")
    
    print("="*70)
    
    # Print best performing model for each metric
    print("\nBest Performing Models:")
    print("-"*50)
    
    # Find best model for each metric
    metrics_to_compare = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-Score")
    ]
    
    for metric_key, metric_name in metrics_to_compare:
        best_model = max(metrics.items(), key=lambda x: x[1][metric_key])
        print(f"{metric_name}: {best_model[0]} ({best_model[1][metric_key]:.4f})")
    
    # Save metrics to a CSV file
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df = metrics_df[["accuracy", "precision", "recall", "f1", "f1_weighted"]]
    metrics_df.to_csv("model_comparison_metrics.csv", index=True)
    
    print(f"\nMetrics saved to model_comparison_metrics.csv")
    
    return metrics

if __name__ == "__main__":
    metrics = main()