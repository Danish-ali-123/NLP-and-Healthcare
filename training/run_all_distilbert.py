#!/usr/bin/env python3
"""
Run DistilBERT training for all languages (EN, HI, PA) using the balanced dataset.

This script trains DistilBERT models for each language and generates comprehensive reports.
"""

import subprocess
import sys
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

def train_distilbert(language: str) -> bool:
    """Train DistilBERT for a specific language."""
    print(f"\n" + "="*80)
    print(f"🚀 TRAINING DistilBERT for {language.upper()}")
    print("="*80)
    
    train_script = Path(__file__).parent / "train_jsonl_distilbert.py"
    balanced_dataset = Path(PROJECT_ROOT) / "data" / "processed" / "multi_language_balanced_dataset.csv"
    config_path = Path(PROJECT_ROOT) / "src" / "config" / "base_distilbert.yaml"
    
    cmd = [
        sys.executable,
        str(train_script),
        "--config", str(config_path),
        "--csv_path", str(balanced_dataset),
        "--output_dir", "train_jsonl/experiments",
        "--language", language,
        "--batch_size", "16",
        "--epochs", "5"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )
    
    # Save output to log file
    log_dir = Path(PROJECT_ROOT) / "train_jsonl" / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"distilbert_{language}.log"
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("STDOUT:\n")
        f.write(result.stdout)
        f.write("\nSTDERR:\n")
        f.write(result.stderr)
    
    if result.returncode != 0:
        print(f"❌ Training failed for {language}: {result.stderr[:500]}...")
        print(f"Log saved to: {log_file}")
        return False
    else:
        print(f"✅ Training completed successfully for {language}")
        print(f"Log saved to: {log_file}")
        return True

def evaluate_distilbert(language: str) -> dict:
    """Evaluate DistilBERT model for a specific language."""
    print(f"\n" + "="*80)
    print(f"📊 EVALUATING DistilBERT for {language.upper()}")
    print("="*80)
    
    # Import predictor
    from predict_jsonl_distilbert import RealTimePredictor
    
    # Paths
    checkpoint_dir = Path(PROJECT_ROOT) / "train_jsonl" / "experiments" / "distilbert" / language
    best_ckpt = checkpoint_dir / "best.ckpt"
    config_path = Path(PROJECT_ROOT) / "src" / "config" / "base_distilbert.yaml"
    labels_path = Path(PROJECT_ROOT) / "preprocess_jsonl" / "labels.json"
    balanced_dataset = Path(PROJECT_ROOT) / "data" / "processed" / "multi_language_balanced_dataset.csv"
    
    if not best_ckpt.exists():
        print(f"❌ Checkpoint not found: {best_ckpt}")
        return None
    
    # Initialize predictor
    predictor = RealTimePredictor(
        checkpoint_path=str(checkpoint_dir),
        config_path=str(config_path),
        labels_path=str(labels_path)
    )
    
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
    
    # Run predictions
    results = predictor.predict_with_metrics(texts, true_labels)
    
    # Save evaluation results
    eval_dir = Path(PROJECT_ROOT) / "train_jsonl" / "results" / "distilbert"
    eval_dir.mkdir(exist_ok=True, parents=True)
    eval_file = eval_dir / f"evaluation_{language}.json"
    
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Evaluation results saved to: {eval_file}")
    return results

def generate_report() -> None:
    """Generate comprehensive report for all languages."""
    print(f"\n" + "="*80)
    print("📋 GENERATING COMPREHENSIVE REPORT")
    print("="*80)
    
    # Create results directory
    results_dir = Path(PROJECT_ROOT) / "train_jsonl" / "results" / "distilbert"
    report_dir = Path(PROJECT_ROOT) / "train_jsonl" / "reports"
    report_dir.mkdir(exist_ok=True)
    
    # Collect results for all languages
    languages = ['en', 'hi', 'pa']
    all_results = {}
    
    for lang in languages:
        eval_file = results_dir / f"evaluation_{lang}.json"
        if eval_file.exists():
            with open(eval_file, 'r', encoding='utf-8') as f:
                all_results[lang] = json.load(f)
        else:
            print(f"⚠️  Evaluation file not found for {lang}: {eval_file}")
    
    # Generate summary report
    summary = {}
    for lang, results in all_results.items():
        metrics = results['metrics']['basic_metrics']
        summary[lang] = {
            'f1_macro': metrics.get('f1_macro', 0),
            'auroc': metrics.get('auroc', 0),
            'auprc': metrics.get('auprc', 0),
            'ece': metrics.get('ece', 0),
            'accuracy': metrics.get('accuracy', 0)
        }
    
    # Save summary
    summary_file = report_dir / "distilbert_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Generate text report
    report_file = report_dir / "distilbert_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# DistilBERT Performance Report\n\n")
        f.write("## Summary Across Languages\n\n")
        f.write("| Language | F1-Macro | AUROC | AUPRC | ECE | Accuracy |\n")
        f.write("|----------|----------|-------|-------|-----|----------|\n")
        
        for lang, metrics in summary.items():
            f.write(f"| {lang.upper()} | {metrics['f1_macro']:.4f} | {metrics['auroc']:.4f} | {metrics['auprc']:.4f} | {metrics['ece']:.4f} | {metrics['accuracy']:.4f} |\n")
        
        f.write("\n## Detailed Results\n\n")
        
        for lang, results in all_results.items():
            f.write(f"### {lang.upper()}\n\n")
            
            # Basic metrics
            metrics = results['metrics']['basic_metrics']
            f.write("#### Basic Metrics\n")
            for metric, value in metrics.items():
                if metric != 'loss':
                    f.write(f"- {metric}: {value:.4f}\n")
            
            # Classification report
            f.write("\n#### Classification Report\n")
            class_report = results['metrics']['classification_report']
            for label, class_metrics in class_report.items():
                if isinstance(class_metrics, dict):
                    f.write(f"- {label}:\n")
                    for m, v in class_metrics.items():
                        if m != 'support' or isinstance(v, int):
                            f.write(f"  - {m}: {v:.4f}\n")
                        else:
                            f.write(f"  - {m}: {v}\n")
            f.write("\n")
    
    print(f"✅ Report generated: {report_file}")
    print(f"✅ Summary saved: {summary_file}")

def main():
    """Main function."""
    print("="*80)
    print("🎯 RUNNING DISTILBERT TRAINING FOR ALL LANGUAGES")
    print("="*80)
    
    # Train for all languages
    languages = ['en', 'hi', 'pa']
    success = []
    
    for lang in languages:
        if train_distilbert(lang):
            success.append(lang)
    
    print(f"\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Successfully trained: {[l.upper() for l in success]}")
    print(f"Failed: {[l.upper() for l in languages if l not in success]}")
    
    # Evaluate for all languages
    print(f"\n" + "="*80)
    print("📊 RUNNING EVALUATION FOR ALL LANGUAGES")
    print("="*80)
    
    for lang in success:
        evaluate_distilbert(lang)
    
    # Generate report
    generate_report()
    
    print(f"\n" + "="*80)
    print("✅ ALL TASKS COMPLETED")
    print("="*80)
    print("Results are saved in:")
    print(f"- Training logs: train_jsonl/logs/")
    print(f"- Model checkpoints: train_jsonl/experiments/distilbert/")
    print(f"- Evaluation results: train_jsonl/results/distilbert/")
    print(f"- Reports: train_jsonl/reports/")

if __name__ == "__main__":
    main()
