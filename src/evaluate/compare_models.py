"""
Model comparison utility for thesis evaluation.

This script loads metrics from both fine-tuned transformer models and zero-shot
LLMs, and produces a comparison table showing Macro-F1 scores (primary metric)
for inclusion in the thesis results section.

Supports:
- Fine-tuned models: DistilBERT, IndicBERT-HPA, XLM-R, mDeBERTa
  (metrics saved at: experiments/E###_MODEL/LANG/metrics.json)
- Zero-shot LLMs: GPT-4o-mini, DeepSeek
  (metrics saved at: experiments/E900_gpt4o_zero_shot/metrics.json,
                     experiments/E901_deepseek_zero_shot/metrics.json)

Output:
- Printed comparison table to console
- CSV file for easy import into thesis documents
"""

import json
import os
import glob
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def load_finetuned_metrics(experiments_dir: str = "experiments") -> Dict[str, Dict[str, Any]]:
    """
    Load metrics from fine-tuned models.
    
    Fine-tuned models save metrics per language at:
    experiments/E###_MODEL/LANG/metrics.json
    
    Args:
        experiments_dir: Base directory for experiments
    
    Returns:
        Dictionary mapping model_name -> {overall: macro_f1, per_lang: {lang: macro_f1}}
    """
    results = {}
    
    # Pattern: experiments/E###_MODEL/LANG/metrics.json
    pattern = os.path.join(experiments_dir, "E*", "*", "metrics.json")
    
    for metrics_path in glob.glob(pattern):
        try:
            # Extract experiment ID, model name, and language
            parts = Path(metrics_path).parts
            if len(parts) >= 3:
                exp_id = parts[-3]  # e.g., E001_distilbert_en
                lang = parts[-2]    # e.g., en, hi, pa
                
                # Extract model name from experiment ID
                # Format: E###_MODEL_NAME or E###_MODEL_NAME_LANG
                model_name = exp_id.split('_', 1)[1] if '_' in exp_id else exp_id
                # Remove language suffix if present
                if model_name.endswith('_en') or model_name.endswith('_hi') or model_name.endswith('_pa'):
                    model_name = model_name.rsplit('_', 1)[0]
                
                # Load metrics
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                
                # Extract macro-F1
                macro_f1 = None
                if 'metrics' in metrics and 'summary' in metrics['metrics']:
                    macro_f1 = metrics['metrics']['summary'].get('macro_f1')
                elif 'macro_f1' in metrics:
                    macro_f1 = metrics['macro_f1']
                
                if macro_f1 is not None:
                    # Initialize model entry if not exists
                    if model_name not in results:
                        results[model_name] = {
                            'type': 'finetuned',
                            'per_lang': {},
                            'overall': None,
                            'n_samples': {}
                        }
                    
                    # Store per-language metric
                    results[model_name]['per_lang'][lang] = macro_f1
                    
                    # Store sample count if available
                    if 'n_samples' in metrics:
                        results[model_name]['n_samples'][lang] = metrics['n_samples']
                    
                    logger.debug(f"Loaded {model_name} ({lang}): Macro-F1 = {macro_f1:.4f}")
        
        except Exception as e:
            logger.warning(f"Failed to load metrics from {metrics_path}: {e}")
            continue
    
    # Compute overall macro-F1 as average of per-language F1s
    for model_name, data in results.items():
        if data['per_lang']:
            data['overall'] = sum(data['per_lang'].values()) / len(data['per_lang'])
    
    return results


def load_zeroshot_metrics(experiments_dir: str = "experiments") -> Dict[str, Dict[str, Any]]:
    """
    Load metrics from zero-shot LLMs.
    
    Zero-shot LLMs save metrics at:
    - experiments/E900_gpt4o_zero_shot/metrics.json
    - experiments/E901_deepseek_zero_shot/metrics.json
    
    Args:
        experiments_dir: Base directory for experiments
    
    Returns:
        Dictionary mapping model_name -> {overall: macro_f1, per_lang: {}}
    """
    results = {}
    
    # Known zero-shot experiment directories
    zeroshot_experiments = {
        'E900_gpt4o_zero_shot': 'GPT-4o-mini',
        'E901_deepseek_zero_shot': 'DeepSeek'
    }
    
    for exp_id, model_name in zeroshot_experiments.items():
        metrics_path = os.path.join(experiments_dir, exp_id, "metrics.json")
        
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                
                # Extract macro-F1
                macro_f1 = metrics.get('macro_f1')
                
                if macro_f1 is not None:
                    results[model_name] = {
                        'type': 'zeroshot',
                        'overall': macro_f1,
                        'per_lang': {},  # Zero-shot models evaluate all languages together
                        'n_samples': metrics.get('num_valid_examples', 0)
                    }
                    
                    logger.debug(f"Loaded {model_name}: Macro-F1 = {macro_f1:.4f}")
            
            except Exception as e:
                logger.warning(f"Failed to load metrics from {metrics_path}: {e}")
                continue
    
    return results


def format_model_name(name: str) -> str:
    """Format model name for display."""
    # Map common model names to display names
    display_names = {
        'distilbert': 'DistilBERT',
        'indicbert_hpa': 'IndicBERT-HPA',
        'indicbert-v1': 'IndicBERT-v1',
        'indicbert-v2': 'IndicBERT-v2',
        'xlm_roberta': 'XLM-RoBERTa',
        'xlm-roberta': 'XLM-RoBERTa',
        'mdeberta': 'mDeBERTa',
        'gpt-4o-mini': 'GPT-4o-mini',
        'deepseek': 'DeepSeek'
    }
    
    # Check exact match first
    if name in display_names:
        return display_names[name]
    
    # Check case-insensitive
    name_lower = name.lower()
    for key, display in display_names.items():
        if key.lower() == name_lower:
            return display
    
    # Capitalize first letter of each word
    return ' '.join(word.capitalize() for word in name.replace('_', ' ').replace('-', ' ').split())


def print_comparison_table(
    finetuned_metrics: Dict[str, Dict[str, Any]],
    zeroshot_metrics: Dict[str, Dict[str, Any]],
    include_per_lang: bool = True
):
    """
    Print comparison table to console.
    
    Args:
        finetuned_metrics: Metrics from fine-tuned models
        zeroshot_metrics: Metrics from zero-shot LLMs
        include_per_lang: Whether to include per-language breakdown
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON - Macro-F1 Scores (Primary Metric)")
    print("="*80)
    
    # Determine languages present
    languages = set()
    for model_data in finetuned_metrics.values():
        languages.update(model_data['per_lang'].keys())
    languages = sorted(languages)  # Usually: ['en', 'hi', 'pa']
    
    # Header
    if include_per_lang and languages:
        header = f"{'Model':<25} {'Overall':<10}"
        for lang in languages:
            header += f" {lang.upper():<10}"
        print(header)
    else:
        print(f"{'Model':<25} {'Type':<15} {'Macro-F1':<10}")
    
    print("-"*80)
    
    # Fine-tuned models
    print("\nFine-Tuned Models:")
    for model_name in sorted(finetuned_metrics.keys()):
        data = finetuned_metrics[model_name]
        display_name = format_model_name(model_name)
        
        if include_per_lang and languages:
            overall = data['overall'] if data['overall'] is not None else 0.0
            row = f"{display_name:<25} {overall:<10.4f}"
            for lang in languages:
                f1 = data['per_lang'].get(lang, 0.0) if lang in data['per_lang'] else None
                if f1 is not None:
                    row += f" {f1:<10.4f}"
                else:
                    row += f" {'-':<10}"
            print(row)
        else:
            overall = data['overall'] if data['overall'] is not None else 0.0
            print(f"{display_name:<25} {'Fine-tuned':<15} {overall:<10.4f}")
    
    # Zero-shot models
    if zeroshot_metrics:
        print("\nZero-Shot LLMs:")
        for model_name in sorted(zeroshot_metrics.keys()):
            data = zeroshot_metrics[model_name]
            display_name = format_model_name(model_name)
            overall = data['overall'] if data['overall'] is not None else 0.0
            
            if include_per_lang and languages:
                row = f"{display_name:<25} {overall:<10.4f}"
                # Zero-shot models don't have per-language breakdown
                for lang in languages:
                    row += f" {'-':<10}"
                print(row)
            else:
                print(f"{display_name:<25} {'Zero-shot':<15} {overall:<10.4f}")
    
    print("="*80)
    print(f"\nNote: Macro-F1 is the primary metric for this thesis.")
    if include_per_lang:
        print("Per-language metrics are shown where available.")
    print()


def save_comparison_csv(
    finetuned_metrics: Dict[str, Dict[str, Any]],
    zeroshot_metrics: Dict[str, Dict[str, Any]],
    output_path: str = "experiments/model_comparison.csv",
    include_per_lang: bool = True
):
    """
    Save comparison table as CSV.
    
    Args:
        finetuned_metrics: Metrics from fine-tuned models
        zeroshot_metrics: Metrics from zero-shot LLMs
        output_path: Path to save CSV file
        include_per_lang: Whether to include per-language breakdown
    """
    # Determine languages present
    languages = set()
    for model_data in finetuned_metrics.values():
        languages.update(model_data['per_lang'].keys())
    languages = sorted(languages)
    
    # Prepare output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        if include_per_lang and languages:
            header = ['Model', 'Type', 'Overall_Macro_F1'] + [f'{lang.upper()}_Macro_F1' for lang in languages]
        else:
            header = ['Model', 'Type', 'Macro_F1']
        writer.writerow(header)
        
        # Fine-tuned models
        for model_name in sorted(finetuned_metrics.keys()):
            data = finetuned_metrics[model_name]
            display_name = format_model_name(model_name)
            overall = data['overall'] if data['overall'] is not None else 0.0
            
            if include_per_lang and languages:
                row = [display_name, 'Fine-tuned', f"{overall:.4f}"]
                for lang in languages:
                    f1 = data['per_lang'].get(lang) if lang in data['per_lang'] else None
                    row.append(f"{f1:.4f}" if f1 is not None else "")
                writer.writerow(row)
            else:
                writer.writerow([display_name, 'Fine-tuned', f"{overall:.4f}"])
        
        # Zero-shot models
        for model_name in sorted(zeroshot_metrics.keys()):
            data = zeroshot_metrics[model_name]
            display_name = format_model_name(model_name)
            overall = data['overall'] if data['overall'] is not None else 0.0
            
            if include_per_lang and languages:
                row = [display_name, 'Zero-shot', f"{overall:.4f}"]
                for lang in languages:
                    row.append("")  # No per-language data for zero-shot
                writer.writerow(row)
            else:
                writer.writerow([display_name, 'Zero-shot', f"{overall:.4f}"])
    
    logger.info(f"Saved comparison table to {output_file}")


def main():
    """Main function to compare all models."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compare metrics from fine-tuned and zero-shot models"
    )
    parser.add_argument(
        '--experiments-dir',
        type=str,
        default='experiments',
        help='Base directory for experiments (default: experiments)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='experiments/model_comparison.csv',
        help='Output CSV file path (default: experiments/model_comparison.csv)'
    )
    parser.add_argument(
        '--no-per-lang',
        action='store_true',
        help='Do not include per-language breakdown'
    )
    parser.add_argument(
        '--no-csv',
        action='store_true',
        help='Do not save CSV file, only print table'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Loading metrics from fine-tuned models...")
    finetuned_metrics = load_finetuned_metrics(args.experiments_dir)
    logger.info(f"Loaded metrics from {len(finetuned_metrics)} fine-tuned models")
    
    logger.info("Loading metrics from zero-shot LLMs...")
    zeroshot_metrics = load_zeroshot_metrics(args.experiments_dir)
    logger.info(f"Loaded metrics from {len(zeroshot_metrics)} zero-shot models")
    
    if not finetuned_metrics and not zeroshot_metrics:
        logger.error("No metrics found! Make sure you have run evaluation.")
        return
    
    # Print comparison table
    print_comparison_table(
        finetuned_metrics,
        zeroshot_metrics,
        include_per_lang=not args.no_per_lang
    )
    
    # Save CSV
    if not args.no_csv:
        save_comparison_csv(
            finetuned_metrics,
            zeroshot_metrics,
            output_path=args.output,
            include_per_lang=not args.no_per_lang
        )


if __name__ == "__main__":
    main()
