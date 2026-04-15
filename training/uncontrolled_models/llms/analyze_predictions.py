import json
import os
from collections import Counter

# Define the models to analyze
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

def analyze_language_predictions(predictions, language):
    """Analyze predictions for a specific language."""
    lang_predictions = [p for p in predictions if p["language"] == language]
    
    if not lang_predictions:
        print(f"No predictions found for {language}")
        return
    
    print(f"\n=== Analysis for {language} ===")
    print(f"Total samples: {len(lang_predictions)}")
    
    # Check true labels distribution
    true_labels = [p["true_label"] for p in lang_predictions]
    pred_labels = [p["label"] for p in lang_predictions]
    
    print("\nTrue Labels Distribution:")
    true_counts = Counter(true_labels)
    for label, count in true_counts.items():
        print(f"  {label}: {count} ({count/len(true_labels)*100:.1f}%)")
    
    print("\nPredicted Labels Distribution:")
    pred_counts = Counter(pred_labels)
    for label, count in pred_counts.items():
        print(f"  {label}: {count} ({count/len(pred_labels)*100:.1f}%)")
    
    # Calculate accuracy
    correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
    accuracy = correct / len(true_labels)
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{len(true_labels)})")
    
    # Show some sample predictions
    print("\nSample Predictions (first 5):")
    for i, pred in enumerate(lang_predictions[:5]):
        print(f"\n{i+1}. True: {pred['true_label']}")
        print(f"   Pred: {pred['label']}")
        print(f"   Confidence: {pred['confidence']}")
        print(f"   Text: {pred['text'][:100]}...")

def main():
    """Main function to analyze predictions."""
    results_dir = "results"
    
    for model_key, model_name in MODEL_NAMES.items():
        file_path = os.path.join(results_dir, f"{model_key}_predictions.jsonl")
        
        if os.path.exists(file_path):
            print("\n" + "="*70)
            print(f"Analyzing {model_name}")
            print("="*70)
            
            predictions = load_predictions(file_path)
            
            # Analyze each language
            for lang in ["hi", "pa"]:
                analyze_language_predictions(predictions, lang)
        else:
            print(f"⚠️  {model_name} predictions file not found: {file_path}")

if __name__ == "__main__":
    main()