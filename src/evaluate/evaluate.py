"""
Comprehensive evaluation pipeline for multilingual clinical NLP models.

Implements full evaluation methodology:
- Language-specific evaluation (en / hi / pa)
- Metrics: Macro-F1, Precision/Recall/F1 per class, AUROC, AUPRC
- Calibration: Expected Calibration Error (ECE)
"""

import argparse
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
from tqdm import tqdm

from src.models.metrics import compute_all_metrics
from src.evaluate.calibration import expected_calibration_error, calibration_summary
from src.data.dataset import MultilingualClinicalDataset, DataCollator, load_label_mapping
from src.models.baselines import build_model_and_tokenizer

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_config: Dict[str, Any],
    label_info: Dict[str, Any],
    device: torch.device
) -> Tuple[torch.nn.Module, Any]:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint directory or best.ckpt file
        model_config: Model configuration dictionary
        label_info: Label mapping information
        device: Device to load model on
    
    Returns:
        Tuple of (model, tokenizer)
    """
    # Check if checkpoint_path is a file or directory
    if os.path.isfile(checkpoint_path):
        # Load from checkpoint file
        checkpoint_dir = os.path.dirname(checkpoint_path)
        model_dir = os.path.join(checkpoint_dir, 'model')
    else:
        # Assume it's a directory
        checkpoint_dir = checkpoint_path
        model_dir = os.path.join(checkpoint_dir, 'model')
        checkpoint_file = os.path.join(checkpoint_dir, 'best.ckpt')
        
        # If model directory doesn't exist, try loading from checkpoint file
        if not os.path.exists(model_dir) and os.path.exists(checkpoint_file):
            checkpoint_dir = os.path.dirname(checkpoint_file) if os.path.isfile(checkpoint_file) else checkpoint_file
            model_dir = os.path.join(checkpoint_dir, 'model')
    
    # Try to load from saved model directory (transformers format)
    if os.path.exists(model_dir):
        logger.info(f"Loading model from {model_dir}")
        try:
            # Try loading as IndicBERT-HPA first
            from src.models.indicbert_hpa import IndicBertHPAClassifier
            model = IndicBertHPAClassifier.from_pretrained(model_dir)
            model.to(device)
            model.eval()
            
            # Load tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            
            return model, tokenizer
        except Exception as e:
            logger.warning(f"Could not load as IndicBERT-HPA, trying standard format: {e}")
    
    # Fallback: build model from config and load state dict
    logger.info("Building model from config and loading state dict")
    model, tokenizer = build_model_and_tokenizer(model_config, label_info, device)
    
    # Try to load state dict from checkpoint
    checkpoint_file = os.path.join(checkpoint_dir, 'best.ckpt')
    if os.path.exists(checkpoint_file):
        logger.info(f"Loading state dict from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.eval()
    return model, tokenizer


def run_inference(
    model: torch.nn.Module,
    tokenizer: Any,
    test_dataset: MultilingualClinicalDataset,
    device: torch.device,
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on test dataset and collect predictions.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        test_dataset: Test dataset
        device: Device to run inference on
        batch_size: Batch size for inference
    
    Returns:
        Tuple of (y_true, y_pred, y_proba)
        - y_true: Ground truth labels [n_samples]
        - y_pred: Predicted labels [n_samples]
        - y_proba: Predicted probabilities [n_samples, n_classes]
    """
    from torch.utils.data import DataLoader
    
    data_collator = DataCollator(tokenizer, padding='longest')
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        collate_fn=data_collator
    )
    
    all_true = []
    all_pred = []
    all_proba = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running inference"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Get logits and probabilities
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            # Get predictions
            preds = torch.argmax(logits, dim=-1)
            
            # Collect results
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())
            all_proba.extend(probs.cpu().numpy())
    
    return np.array(all_true), np.array(all_pred), np.array(all_proba)


def evaluate_language(
    language: str,
    experiment_dir: str,
    model_config: Dict[str, Any],
    base_config: Dict[str, Any],
    label_info: Dict[str, Any],
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate model for a specific language.
    
    Args:
        language: Language code ('en', 'hi', or 'pa')
        experiment_dir: Experiment directory (e.g., experiments/E001_distilbert_en)
        model_config: Model configuration
        base_config: Base configuration
        label_info: Label mapping information
        device: Device to run evaluation on
    
    Returns:
        Dictionary containing all evaluation results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating language: {language.upper()}")
    logger.info(f"{'='*60}\n")
    
    # Language-specific directory
    lang_dir = os.path.join(experiment_dir, language)
    
    if not os.path.exists(lang_dir):
        logger.warning(f"Language directory not found: {lang_dir}")
        return None
    
    # Find checkpoint
    checkpoint_path = os.path.join(lang_dir, 'best.ckpt')
    if not os.path.exists(checkpoint_path):
        # Try model directory
        model_dir = os.path.join(lang_dir, 'model')
        if os.path.exists(model_dir):
            checkpoint_path = model_dir
        else:
            logger.error(f"No checkpoint found in {lang_dir}")
            return None
    
    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    try:
        model, tokenizer = load_model_from_checkpoint(
            checkpoint_path, model_config, label_info, device
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None
    
    # Load test data
    processed_dir = base_config['paths']['data_processed']
    test_path = os.path.join(processed_dir, 'test.csv')
    
    if not os.path.exists(test_path):
        logger.error(f"Test data not found: {test_path}")
        return None
    
    # Get max sequence length
    max_seq_len = base_config.get('data', {}).get('max_seq_len', 256)
    if 'model' in model_config and 'tokenizer' in model_config['model']:
        max_seq_len = model_config['model']['tokenizer'].get('max_length', max_seq_len)
    
    # Create test dataset with language filtering
    test_dataset = MultilingualClinicalDataset(
        test_path,
        tokenizer,
        label_info['label2id'],
        max_length=max_seq_len,
        language=language
    )
    
    if len(test_dataset) == 0:
        logger.warning(f"No test samples found for language {language}")
        return None
    
    logger.info(f"Test samples for {language}: {len(test_dataset)}")
    
    # Run inference
    logger.info("Running inference on test set...")
    y_true, y_pred, y_proba = run_inference(
        model, tokenizer, test_dataset, device, batch_size=32
    )
    
    # Get label names for metrics
    label_names = [label_info['id2label'][i] for i in sorted(label_info['id2label'].keys())]
    
    # Compute all metrics
    logger.info("Computing metrics...")
    metrics = compute_all_metrics(y_true, y_pred, y_proba, label_names)
    
    # Compute calibration
    logger.info("Computing calibration...")
    ece = expected_calibration_error(y_true, y_proba, n_bins=15)
    calibration = calibration_summary(y_true, y_proba, n_bins=15)
    
    # Combine results
    results = {
        'language': language,
        'n_samples': len(y_true),
        'metrics': metrics,
        'calibration': {
            'ece': ece,
            'summary': calibration
        }
    }
    
    # Save results
    metrics_path = os.path.join(lang_dir, 'metrics.json')
    os.makedirs(lang_dir, exist_ok=True)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    calibration_path = os.path.join(lang_dir, 'calibration.json')
    with open(calibration_path, 'w', encoding='utf-8') as f:
        json.dump(calibration, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved calibration to {calibration_path}")
    
    return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained models on test set',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a specific experiment
  python -m src.evaluate.evaluate --exp experiments/E001_distilbert_en
  
  # Evaluate with custom config
  python -m src.evaluate.evaluate --exp experiments/E001_distilbert_en --config src/config/base.yaml
        """
    )
    parser.add_argument(
        '--exp', '--experiment',
        type=str,
        required=True,
        help='Experiment directory (e.g., experiments/E001_distilbert_en)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='src/config/base.yaml',
        help='Path to base configuration file (default: src/config/base.yaml)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, default: auto)'
    )
    parser.add_argument(
        '--languages',
        type=str,
        nargs='+',
        default=None,
        help='Languages to evaluate (default: all from config)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    
    # Load configurations
    import yaml
    with open(args.config, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    # Determine which languages to evaluate
    if args.languages:
        languages = args.languages
    else:
        languages = base_config.get('data', {}).get('languages', ['en', 'hi', 'pa'])
    
    # Load label info
    processed_dir = base_config['paths']['data_processed']
    labels_path = os.path.join(processed_dir, 'labels.json')
    label_info = load_label_mapping(labels_path)
    
    # Determine model config based on language
    # English → DistilBERT, Hindi/Punjabi → IndicBERT-HPA
    experiment_dir = args.exp
    
    # Check if experiment directory exists
    if not os.path.exists(experiment_dir):
        logger.error(f"Experiment directory not found: {experiment_dir}")
        return
    
    # Load model configs
    distilbert_config_path = 'src/config/base_distilbert.yaml'
    indicbert_hpa_config_path = 'src/config/base_indicbert_hpa.yaml'
    
    distilbert_config = {}
    indicbert_hpa_config = {}
    
    if os.path.exists(distilbert_config_path):
        with open(distilbert_config_path, 'r', encoding='utf-8') as f:
            distilbert_config = yaml.safe_load(f)
    
    if os.path.exists(indicbert_hpa_config_path):
        with open(indicbert_hpa_config_path, 'r', encoding='utf-8') as f:
            indicbert_hpa_config = yaml.safe_load(f)
    
    # Evaluate each language
    all_results = {}
    
    for lang in languages:
        # Select appropriate model config
        if lang == 'en':
            model_config = distilbert_config
        else:  # hi or pa
            model_config = indicbert_hpa_config
        
        # Merge with base config
        merged_config = base_config.copy()
        if 'training' in model_config:
            if 'training' not in merged_config:
                merged_config['training'] = {}
            merged_config['training'].update(model_config['training'])
        if 'model' in model_config:
            merged_config['model'] = model_config['model']
        
        # Evaluate
        results = evaluate_language(
            language=lang,
            experiment_dir=experiment_dir,
            model_config=merged_config,
            base_config=base_config,
            label_info=label_info,
            device=device
        )
        
        if results:
            all_results[lang] = results
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Language':<10} {'Macro-F1':<12} {'AUROC (macro)':<15} {'AUPRC (macro)':<15} {'ECE':<10}")
    print("-"*80)
    
    for lang, results in all_results.items():
        if results:
            metrics = results['metrics']
            summary = metrics['summary']
            ece = results['calibration']['ece']
            
            macro_f1 = summary.get('macro_f1', 0.0)
            macro_auroc = summary.get('macro_auroc', 0.0)
            macro_auprc = summary.get('macro_auprc', 0.0)
            
            print(f"{lang.upper():<10} {macro_f1:<12.4f} {macro_auroc:<15.4f} {macro_auprc:<15.4f} {ece:<10.4f}")
    
    print("="*80)
    
    # Save overall summary
    summary_path = os.path.join(experiment_dir, 'evaluation_summary.json')
    summary_data = {
        'experiment': experiment_dir,
        'languages': list(all_results.keys()),
        'results': {
            lang: {
                'macro_f1': results['metrics']['summary']['macro_f1'],
                'macro_auroc': results['metrics']['summary']['macro_auroc'],
                'macro_auprc': results['metrics']['summary']['macro_auprc'],
                'ece': results['calibration']['ece'],
                'n_samples': results['n_samples']
            }
            for lang, results in all_results.items() if results
        }
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSaved evaluation summary to {summary_path}")


if __name__ == '__main__':
    main()
