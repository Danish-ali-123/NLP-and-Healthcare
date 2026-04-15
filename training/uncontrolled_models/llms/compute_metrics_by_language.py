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

def compute_model_metrics(predictions, language=None):
    """Compute metrics for a single model's predictions, optionally filtered by language."""
    # Filter by language if specified
    if language:
        predictions = [p for p in predictions if p["language"] == language]
    
    y_true = []
    y_pred = []
    
    for pred in predictions:
        y_true.append(pred["true_label"])
        y_pred.append(pred["label"])
    
    if not y_true:
        return None
    
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
        "y_pred": y_pred,
        "support": len(y_true)
    }

def main():
    """Main function to compute metrics by language for all models."""
    results_dir = "results"
    all_metrics = {}
    
    print("=== LLM Model Evaluation Metrics by Language ===")
    print("Computing metrics for each model and language combination...\n")
    
    # Process each model
    for model_key, model_name in MODEL_NAMES.items():
        file_path = os.path.join(results_dir, f"{model_key}_predictions.jsonl")
        
        if os.path.exists(file_path):
            print(f"Processing {model_name}...")
            predictions = load_predictions(file_path)
            
            # Get all unique languages in the predictions
            languages = list(set([p["language"] for p in predictions]))
            languages.sort()
            
            model_metrics = {}
            
            # Compute metrics for each language
            for lang in languages:
                print(f"  - {lang}...")
                lang_metrics = compute_model_metrics(predictions, language=lang)
                if lang_metrics:
                    model_metrics[lang] = lang_metrics
            
            # Compute overall metrics too
            print(f"  - Overall...")
            overall_metrics = compute_model_metrics(predictions)
            if overall_metrics:
                model_metrics["overall"] = overall_metrics
            
            all_metrics[model_name] = model_metrics
        else:
            print(f"⚠️  {model_name} predictions file not found: {file_path}")
    
    # Print detailed metrics by language for each model
    for model_name, model_metrics in all_metrics.items():
        print(f"\n" + "="*70)
        print(f"{model_name} - Metrics by Language")
        print("="*70)
        print(f"{'Language':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>8}")
        print("-"*70)
        
        for lang, metrics in model_metrics.items():
            print(f"{lang:<15} "
                  f"{metrics['accuracy']:<10.4f} "
                  f"{metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} "
                  f"{metrics['f1']:<10.4f} "
                  f"{metrics['support']:>8}")
    
    # Create a comparison dataframe
    comparison_data = []
    for model_name, model_metrics in all_metrics.items():
        for lang, metrics in model_metrics.items():
            comparison_data.append({
                "model": model_name,
                "language": lang,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "f1_weighted": metrics["f1_weighted"],
                "support": metrics["support"]
            })
    
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    df.to_csv("model_metrics_by_language.csv", index=False)
    print(f"\nMetrics saved to model_metrics_by_language.csv")
    
    # Create a pivot table for better comparison
    pivot_df = df.pivot_table(
        index="language",
        columns="model",
        values=["accuracy", "precision", "recall", "f1"],
        aggfunc="first"
    )
    
    print("\n" + "="*100)
    print("Model Comparison by Language")
    print("="*100)
    print(pivot_df.round(4))
    
    # Save pivot table to CSV
    pivot_df.to_csv("model_comparison_by_language.csv")
    print(f"\nComparison table saved to model_comparison_by_language.csv")
    
    return all_metrics

if __name__ == "__main__":
    metrics = main()