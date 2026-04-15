"""
Hyperparameter sweep script for IndicBERT-HPA.

This script runs a grid search over hyperparameter combinations and logs results
to a CSV file for easy comparison.

Usage:
    python scripts/train/run_indicbert_hpa_sweep.py
"""

import subprocess
import sys
import yaml
import tempfile
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from itertools import product

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Base config path
BASE_CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "base_indicbert_hpa.yaml"


def load_base_config() -> Dict[str, Any]:
    """Load the base IndicBERT-HPA config."""
    with open(BASE_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_variant_config(
    base_config: Dict[str, Any],
    variant: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a config variant by overriding base config with variant parameters.
    
    Args:
        base_config: Base configuration dictionary
        variant: Variant parameters to override
        
    Returns:
        Modified configuration dictionary
    """
    config = yaml.safe_load(yaml.dump(base_config))  # Deep copy
    
    # Override languages to train only Hindi and Punjabi (skip English)
    if 'data' not in config:
        config['data'] = {}
    config['data']['languages'] = ['hi', 'pa']
    
    # Override training hyperparameters
    if 'batch_size' in variant:
        config['training']['batch_size'] = variant['batch_size']
    if 'learning_rate' in variant:
        config['training']['learning_rate'] = variant['learning_rate']
    if 'epochs' in variant:
        config['training']['epochs'] = variant['epochs']
    if 'optimizer' in variant:
        config['training']['optimizer'] = variant['optimizer']
    if 'early_stopping_monitor' in variant:
        if 'early_stopping' not in config['training']:
            config['training']['early_stopping'] = {}
        config['training']['early_stopping']['monitor'] = variant['early_stopping_monitor']
    
    # Override HPA architecture for Hindi and Punjabi
    if 'hidden_sizes' in variant:
        if 'hi' in config.get('models', {}):
            if 'hpa' not in config['models']['hi']:
                config['models']['hi']['hpa'] = {}
            config['models']['hi']['hpa']['hidden_sizes'] = variant['hidden_sizes']
        if 'pa' in config.get('models', {}):
            if 'hpa' not in config['models']['pa']:
                config['models']['pa']['hpa'] = {}
            config['models']['pa']['hpa']['hidden_sizes'] = variant['hidden_sizes']
    
    if 'dropout' in variant:
        if 'hi' in config.get('models', {}):
            if 'hpa' not in config['models']['hi']:
                config['models']['hi']['hpa'] = {}
            config['models']['hi']['hpa']['dropout'] = variant['dropout']
        if 'pa' in config.get('models', {}):
            if 'hpa' not in config['models']['pa']:
                config['models']['pa']['hpa'] = {}
            config['models']['pa']['hpa']['dropout'] = variant['dropout']
    
    return config


def extract_metrics_from_experiment(experiment_dir: Path, variant_name: str) -> Optional[Dict[str, Any]]:
    """
    Extract metrics from metrics_best.json files (standard location).
    
    Args:
        experiment_dir: Experiment directory (e.g., experiments/indicbert_hpa_sweep/variant_name/E<timestamp>)
        variant_name: Name of the variant
    
    Returns:
        Dictionary with metrics per language, or None if metrics files are missing
    """
    metrics = {
        'variant_name': variant_name,
        'experiment_dir': str(experiment_dir)
    }
    
    # Check if experiment directory exists
    if not experiment_dir.exists():
        print(f"⚠️  Warning: Experiment directory does not exist: {experiment_dir}")
        return None
    
    # Find experiment ID subdirectory (e.g., E1764396596)
    try:
        experiment_id_dirs = [d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith('E')]
    except Exception as e:
        print(f"⚠️  Warning: Could not list experiment directory {experiment_dir}: {e}")
        return None
    
    if not experiment_id_dirs:
        print(f"⚠️  Warning: No experiment ID directory found in {experiment_dir}")
        return None
    
    # Use the most recent experiment ID directory (in case of multiple)
    experiment_id_dir = max(experiment_id_dirs, key=lambda p: p.stat().st_mtime)
    
    # Look for metrics_best.json in language subdirectories (hi and pa only)
    missing_languages = []
    for lang in ['hi', 'pa']:
        lang_dir = experiment_id_dir / lang
        metrics_best_path = lang_dir / 'metrics_best.json'
        
        if not metrics_best_path.exists():
            missing_languages.append(lang)
            print(f"⚠️  Warning: metrics_best.json not found for {lang} at {metrics_best_path}")
            continue
        
        try:
            with open(metrics_best_path, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            
            # Extract best metrics
            metrics[f'{lang}_best_val_loss'] = metrics_data.get('best_val_loss', None)
            metrics[f'{lang}_best_val_accuracy'] = metrics_data.get('best_val_accuracy', None)
            metrics[f'{lang}_best_val_f1_macro'] = metrics_data.get('best_val_f1_macro', None)
            metrics[f'{lang}_best_epoch'] = metrics_data.get('best_epoch', None)
            
            print(f"✓ Loaded metrics for {lang}: F1={metrics_data.get('best_val_f1_macro', 'N/A'):.4f}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load metrics_best.json for {lang}: {e}")
            missing_languages.append(lang)
    
    # If any required language is missing, return None
    if missing_languages:
        print(f"❌ Missing metrics for languages: {missing_languages}")
        return None
    
    return metrics


def run_variant(variant: Dict[str, Any], variant_idx: int, total_variants: int, 
                results_csv_path: Path) -> Optional[Dict[str, Any]]:
    """
    Run training for a single variant and extract metrics.
    
    Args:
        variant: Variant configuration dictionary
        variant_idx: Index of current variant (1-based)
        total_variants: Total number of variants
        results_csv_path: Path to CSV file for logging results
    
    Returns:
        Dictionary with metrics, or None if training failed
    """
    variant_name = variant.get('name', f'variant_{variant_idx}')
    
    print("\n" + "=" * 80)
    print(f"🔬 VARIANT {variant_idx}/{total_variants}: {variant_name}")
    print("=" * 80)
    print(f"Parameters:")
    for key, value in variant.items():
        if key != 'name':
            print(f"  {key}: {value}")
    print("=" * 80 + "\n")
    
    # Load base config
    base_config = load_base_config()
    
    # Create variant config
    variant_config = create_variant_config(base_config, variant)
    
    # Set experiment root to sweep subdirectory
    # Structure: experiments/indicbert_hpa_sweep/variant_name/
    variant_config['training']['experiment_root'] = f"experiments/indicbert_hpa_sweep/{variant_name}"
    
    # Create tmp directory if needed
    tmp_dir = PROJECT_ROOT / 'tmp'
    tmp_dir.mkdir(exist_ok=True)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, 
                                     dir=str(tmp_dir)) as f:
        yaml.dump(variant_config, f, default_flow_style=False, sort_keys=False)
        temp_config_path = f.name
    
    try:
        # Run training
        result = subprocess.run(
            [
                sys.executable,
                "-m", "src.train.train",
                "--config", temp_config_path,
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ Training failed for variant {variant_name} (exit code: {result.returncode})")
            if result.stderr:
                print("=" * 80)
                print("FULL STDERR OUTPUT:")
                print("=" * 80)
                print(result.stderr)
                print("=" * 80)
            if result.stdout:
                print("=" * 80)
                print("STDOUT OUTPUT (last 1000 chars):")
                print("=" * 80)
                print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
                print("=" * 80)
            return None
        else:
            print(f"✅ Training completed for variant {variant_name}")
            
            # Extract metrics from experiment directory
            experiment_dir = PROJECT_ROOT / variant_config['training']['experiment_root']
            metrics = extract_metrics_from_experiment(experiment_dir, variant_name)
            
            if metrics:
                # Add variant parameters to metrics
                for key, value in variant.items():
                    if key != 'name':
                        metrics[f'param_{key}'] = str(value) if not isinstance(value, list) else ','.join(map(str, value))
                
                return metrics
            else:
                print(f"❌ Failed to extract metrics for {variant_name} - missing metrics_best.json files")
                # Return a failure record with variant params
                failure_metrics = {
                    'variant_name': variant_name,
                    'experiment_dir': str(experiment_dir),
                    'status': 'FAILED_METRICS_MISSING'
                }
                for key, value in variant.items():
                    if key != 'name':
                        failure_metrics[f'param_{key}'] = str(value) if not isinstance(value, list) else ','.join(map(str, value))
                return failure_metrics
            
    except Exception as e:
        print(f"❌ Error running variant {variant_name}: {str(e)}")
        return None
    finally:
        # Clean up temporary config file
        try:
            Path(temp_config_path).unlink()
        except:
            pass


def generate_variant_name(batch_size: int, learning_rate: float, hidden_sizes: List[int], 
                          dropout: float) -> str:
    """Generate a short, descriptive variant name."""
    h_str = '_'.join(map(str, hidden_sizes))
    lr_str = f"{learning_rate:.0e}".replace('e-0', 'e-').replace('e+', 'e')
    return f"bs{batch_size}_lr{lr_str}_h{h_str}_do{dropout}"


def main():
    """Main function to run hyperparameter sweep with grid search."""
    
    # Check for test mode (only 2 variants)
    import sys
    test_mode = '--test' in sys.argv or '--quick' in sys.argv
    
    if test_mode:
        print("🧪 TEST MODE: Running only 2 variants for quick verification")
        # Define minimal grid search space for testing
        batch_sizes = [32]
        learning_rates = [3e-5]
        epochs = [2]  # Very short for testing
        early_stopping_monitor = ['val_f1_macro']
        hidden_sizes_list = [[512]]
        dropout_values = [0.2]
    else:
        # Define full grid search space
        batch_sizes = [16, 32]
        learning_rates = [2e-5, 3e-5, 5e-5]
        epochs = [20]
        early_stopping_monitor = ['val_f1_macro']
        hidden_sizes_list = [[512], [768, 384]]
        dropout_values = [0.1, 0.2, 0.3]
    
    # Generate all combinations
    all_combinations = list(product(
        batch_sizes,
        learning_rates,
        epochs,
        early_stopping_monitor,
        hidden_sizes_list,
        dropout_values
    ))
    
    if test_mode:
        # Limit to 2 variants for testing
        all_combinations = all_combinations[:2]
        print(f"   Limited to {len(all_combinations)} variants for testing")
    
    # Create variant dictionaries
    hpa_variants: List[Dict[str, Any]] = []
    for bs, lr, ep, monitor, h_sizes, do in all_combinations:
        variant_name = generate_variant_name(bs, lr, h_sizes, do)
        hpa_variants.append({
            "name": variant_name,
            "batch_size": bs,
            "learning_rate": lr,
            "epochs": ep,
            "early_stopping_monitor": monitor,
            "hidden_sizes": h_sizes,
            "dropout": do
        })
    
    print(f"📦 Project root: {PROJECT_ROOT}")
    print(f"📋 Base config: {BASE_CONFIG_PATH}")
    print(f"🔬 Grid search: {len(hpa_variants)} combinations")
    print(f"   batch_size: {batch_sizes}")
    print(f"   learning_rate: {learning_rates}")
    print(f"   epochs: {epochs}")
    print(f"   early_stopping.monitor: {early_stopping_monitor}")
    print(f"   hidden_sizes: {hidden_sizes_list}")
    print(f"   dropout: {dropout_values}\n")
    
    # Create results CSV file
    reports_dir = PROJECT_ROOT / 'reports'
    reports_dir.mkdir(exist_ok=True)
    results_csv_path = reports_dir / 'hpa_sweep_results.csv'
    
    # Prepare CSV header
    csv_columns = [
        'variant_name', 'experiment_dir', 'status',
        'param_batch_size', 'param_learning_rate', 'param_epochs',
        'param_early_stopping_monitor', 'param_hidden_sizes', 'param_dropout',
        'hi_best_val_loss', 'hi_best_val_accuracy', 'hi_best_val_f1_macro', 'hi_best_epoch',
        'pa_best_val_loss', 'pa_best_val_accuracy', 'pa_best_val_f1_macro', 'pa_best_epoch',
    ]
    
    # Initialize CSV file with header (only if file doesn't exist)
    if not results_csv_path.exists():
        with open(results_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
        print(f"📊 Created new results file: {results_csv_path}\n")
    else:
        print(f"📊 Appending to existing results file: {results_csv_path}\n")
    
    # Create tmp directory if it doesn't exist
    tmp_dir = PROJECT_ROOT / 'tmp'
    tmp_dir.mkdir(exist_ok=True)
    
    # Run each variant and collect results
    all_results = []
    for idx, variant in enumerate(hpa_variants, 1):
        metrics = run_variant(variant, idx, len(hpa_variants), results_csv_path)
        
        if metrics:
            # Add status if not present
            if 'status' not in metrics:
                metrics['status'] = 'SUCCESS'
            
            all_results.append(metrics)
            
            # Append to CSV immediately (in case of interruption)
            with open(results_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns)
                # Only write columns that exist in metrics, fill missing with empty string
                row = {col: metrics.get(col, '') for col in csv_columns}
                writer.writerow(row)
            
            print(f"✓ Saved results for {variant['name']} to CSV\n")
    
    print("\n" + "=" * 80)
    print("✅ Hyperparameter sweep completed!")
    print("=" * 80)
    print(f"\n📊 Results saved to: {results_csv_path}")
    print(f"📁 Experiments saved in: {PROJECT_ROOT / 'experiments' / 'indicbert_hpa_sweep'}")
    
    # Print summary
    successful_results = [r for r in all_results if r.get('status') == 'SUCCESS']
    failed_results = [r for r in all_results if r.get('status') != 'SUCCESS']
    
    print(f"\n📊 Summary: {len(successful_results)} successful, {len(failed_results)} failed out of {len(all_results)} total")
    
    if failed_results:
        print(f"\n❌ Failed variants:")
        for result in failed_results:
            print(f"  - {result['variant_name']}: {result.get('status', 'UNKNOWN')}")
    
    # Print summary of best models
    if successful_results:
        print("\n🏆 Top 5 models by Hindi F1-macro:")
        sorted_by_hi_f1 = sorted(
            [r for r in successful_results if r.get('hi_best_val_f1_macro') is not None and r.get('hi_best_val_f1_macro') != ''],
            key=lambda x: float(x.get('hi_best_val_f1_macro', 0)) if isinstance(x.get('hi_best_val_f1_macro'), (int, float)) else 0,
            reverse=True
        )[:5]
        for i, result in enumerate(sorted_by_hi_f1, 1):
            hi_f1 = result.get('hi_best_val_f1_macro', 'N/A')
            pa_f1 = result.get('pa_best_val_f1_macro', 'N/A')
            print(f"  {i}. {result['variant_name']}: "
                  f"hi_f1={hi_f1:.4f if isinstance(hi_f1, (int, float)) else hi_f1}, "
                  f"pa_f1={pa_f1:.4f if isinstance(pa_f1, (int, float)) else pa_f1}")
        
        print("\n🏆 Top 5 models by Punjabi F1-macro:")
        sorted_by_pa_f1 = sorted(
            [r for r in successful_results if r.get('pa_best_val_f1_macro') is not None and r.get('pa_best_val_f1_macro') != ''],
            key=lambda x: float(x.get('pa_best_val_f1_macro', 0)) if isinstance(x.get('pa_best_val_f1_macro'), (int, float)) else 0,
            reverse=True
        )[:5]
        for i, result in enumerate(sorted_by_pa_f1, 1):
            hi_f1 = result.get('hi_best_val_f1_macro', 'N/A')
            pa_f1 = result.get('pa_best_val_f1_macro', 'N/A')
            print(f"  {i}. {result['variant_name']}: "
                  f"hi_f1={hi_f1:.4f if isinstance(hi_f1, (int, float)) else hi_f1}, "
                  f"pa_f1={pa_f1:.4f if isinstance(pa_f1, (int, float)) else pa_f1}")


if __name__ == "__main__":
    main()

