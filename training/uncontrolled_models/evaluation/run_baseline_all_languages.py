#!/usr/bin/env python3
"""
Run baseline DistilBERT model for all languages (EN, HI, PA).

This script runs the baseline model (pretrained encoder + random head, no training)
for all languages and generates comprehensive results with standard deviation.
"""

import sys
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

def run_baseline_for_language(
    language: str,
    num_runs: int = 5
) -> dict:
    """Run baseline predictions for a language."""
    print(f"\n" + "="*80)
    print(f"📊 RUNNING BASELINE DistilBERT for {language.upper()}")
    print("="*80)
    
    # Import predictor
    from predict_jsonl_distilbert import RealTimePredictor
    
    # Paths
    checkpoint_dir = Path(PROJECT_ROOT) / "train_jsonl" / "experiments" / "distilbert" / language
    config_path = Path(PROJECT_ROOT) / "src" / "config" / "base_distilbert.yaml"
    balanced_dataset = Path(PROJECT_ROOT) / "uncontrolled_models" / "raw" / "multilingual_dataset.csv"
    
    # Load balanced dataset
    df = pd.read_csv(balanced_dataset)
    
    # Filter by language
    if language == 'en':
        # English data has NaN in language column
        lang_df = df[df['language'].isna()]
    else:
        lang_df = df[df['language'] == language]
    
    print(f"Evaluating on {len(lang_df)} samples")
    
    # Get texts and true labels
    texts = lang_df['text_input'].tolist()
    true_labels = lang_df['Diagnosis Category'].tolist()
    
    # Generate language-specific label mapping
    unique_labels = sorted(list(set(true_labels)))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {str(i): label for label, i in label2id.items()}
    label_info = {
        'labels': unique_labels,
        'label2id': label2id,
        'id2label': id2label,
        'num_labels': len(unique_labels)
    }
    
    # Create temporary labels file for this language
    temp_labels_path = Path(PROJECT_ROOT) / f"temp_labels_{language}.json"
    with open(temp_labels_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(label_info, f, ensure_ascii=False, indent=2)
    
    print(f"Generated language-specific labels: {unique_labels}")
    
    results = []
    
    for run in range(num_runs):
        print(f"\n--- Run {run+1}/{num_runs} ---")
        
        # Initialize predictor with baseline mode (no_ckpt=True)
        predictor = RealTimePredictor(
            checkpoint_path=str(checkpoint_dir),
            config_path=str(config_path),
            labels_path=str(temp_labels_path),
            no_ckpt=True  # Baseline mode
        )
        
        # Run predictions
        run_results = predictor.predict_with_metrics(texts, true_labels)
        results.append(run_results)
    
    # Clean up temporary labels file
    if temp_labels_path.exists():
        temp_labels_path.unlink()
    
    return results

def calculate_statistics(results: list) -> dict:
    """Calculate statistics from multiple runs."""
    if not results:
        return {}
    
    # Extract metrics from each run
    metrics_list = []
    for run_result in results:
        metrics_list.append(run_result['metrics']['basic_metrics'])
    
    # Calculate mean and std for each metric
    stats = {}
    metrics = metrics_list[0].keys()
    
    for metric in metrics:
        values = [m.get(metric, 0) for m in metrics_list]
        stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    return stats

def generate_report() -> None:
    """Generate comprehensive report for all languages."""
    print(f"\n" + "="*80)
    print("📋 GENERATING COMPREHENSIVE BASELINE REPORT")
    print("="*80)
    
    # Create results directory
    results_dir = Path(PROJECT_ROOT) / "train_jsonl" / "results" / "baseline"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Run baseline for all languages
    languages = ['en', 'hi', 'pa']
    all_results = {}
    all_stats = {}
    
    for lang in languages:
        print(f"\n" + "="*60)
        print(f"Processing {lang.upper()}")
        print("="*60)
        
        # Run baseline predictions
        run_results = run_baseline_for_language(lang, num_runs=5)
        
        # Calculate statistics
        stats = calculate_statistics(run_results)
        
        all_results[lang] = run_results
        all_stats[lang] = stats
        
        # Save results
        results_file = results_dir / f"baseline_results_{lang}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'results': run_results,
                'statistics': stats
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Results saved to: {results_file}")
    
    # Generate summary report
    summary_file = results_dir / "baseline_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    
    # Generate text report
    report_file = results_dir / "baseline_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# DistilBERT Baseline Performance Report\n\n")
        f.write("## Summary Across Languages (Mean ± Std)\n\n")
        f.write("| Language | F1-Macro | Accuracy | Balanced Accuracy | MCC |\n")
        f.write("|----------|----------|----------|-------------------|-----|\n")
        
        for lang, stats in all_stats.items():
            f.write(f"| {lang.upper()} | ")
            f.write(f"{stats['f1_macro']['mean']:.4f} ± {stats['f1_macro']['std']:.4f} | ")
            f.write(f"{stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f} | ")
            f.write(f"{stats['balanced_accuracy']['mean']:.4f} ± {stats['balanced_accuracy']['std']:.4f} | ")
            f.write(f"{stats['mcc']['mean']:.4f} ± {stats['mcc']['std']:.4f} |\n")
        
        f.write("\n## Detailed Results\n\n")
        
        for lang, stats in all_stats.items():
            f.write(f"### {lang.upper()}\n\n")
            f.write("#### Metrics (Mean ± Std)\n")
            for metric, values in stats.items():
                f.write(f"- {metric}: {values['mean']:.4f} ± {values['std']:.4f}\n")
            f.write("\n")
    
    print(f"\n✅ Summary saved: {summary_file}")
    print(f"✅ Report generated: {report_file}")

def main():
    """Main function."""
    print("="*80)
    print("🎯 RUNNING DISTILBERT BASELINE FOR ALL LANGUAGES")
    print("="*80)
    print("Baseline mode: pretrained encoder + random head, no training")
    print("Calculating Mean ± Std over 5 runs")
    print("="*80)
    
    generate_report()
    
    print(f"\n" + "="*80)
    print("✅ ALL BASELINE RESULTS GENERATED")
    print("="*80)
    print("Results are saved in:")
    print(f"- Baseline results: train_jsonl/results/baseline/")

if __name__ == "__main__":
    main()
