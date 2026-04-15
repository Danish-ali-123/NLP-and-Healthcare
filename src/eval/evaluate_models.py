"""
Generic evaluation script for trained models on test/val/train splits.

Reuses the same model building and metrics logic as training to ensure consistency.
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import MultilingualClinicalDataset, DataCollator, load_label_mapping
from src.models.baselines import build_model_and_tokenizer
from src.models.metrics import MetricCalculator
from src.evaluate.evaluate import load_model_from_checkpoint
from src.train.train import load_config, build_model_config_from_models_section

logger = logging.getLogger(__name__)


def find_latest_experiment(experiment_root: str) -> Optional[str]:
    """
    Find the most recent experiment ID in the experiment root directory.
    
    Args:
        experiment_root: Root directory containing experiment subdirectories
        
    Returns:
        Most recent experiment ID (e.g., "E1764404021") or None if no experiments found
    """
    if not os.path.exists(experiment_root):
        return None
    
    experiment_dirs = [
        d for d in os.listdir(experiment_root)
        if os.path.isdir(os.path.join(experiment_root, d)) and d.startswith('E')
    ]
    
    if not experiment_dirs:
        return None
    
    # Sort by experiment ID (assuming E followed by timestamp)
    experiment_dirs.sort(reverse=True)
    return experiment_dirs[0]


def load_trained_model_for_eval(
    model_config: Dict[str, Any],
    label_info: Dict[str, Any],
    checkpoint_dir: str,
    device: torch.device
) -> Tuple[torch.nn.Module, Any]:
    """
    Load a trained model from checkpoint directory for evaluation.
    
    Args:
        model_config: Model configuration dict (compatible with build_model_and_tokenizer)
        label_info: Label mapping information
        checkpoint_dir: Directory containing best.ckpt or model/ subdirectory
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer) with model in eval mode
    """
    checkpoint_path = os.path.join(checkpoint_dir, 'best.ckpt')
    
    if not os.path.exists(checkpoint_path):
        # Try model directory
        model_dir = os.path.join(checkpoint_dir, 'model')
        if os.path.exists(model_dir):
            checkpoint_path = model_dir
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {checkpoint_dir}. "
                f"Expected best.ckpt or model/ subdirectory."
            )
    
    model, tokenizer = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        label_info=label_info,
        device=device
    )
    
    model.eval()
    return model, tokenizer


def evaluate_language(
    language: str,
    model_config: Dict[str, Any],
    base_config: Dict[str, Any],
    label_info: Dict[str, Any],
    split: str,
    experiment_dir: str,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate a trained model for a specific language on a data split.
    
    Args:
        language: Language code ('en', 'hi', or 'pa')
        model_config: Model configuration dict
        base_config: Base configuration
        label_info: Label mapping information
        split: Data split ('train', 'val', or 'test')
        experiment_dir: Experiment directory containing language subdirectories
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing evaluation results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating language: {language.upper()} on split: {split.upper()}")
    logger.info(f"{'='*60}\n")
    
    # Get language-specific directory
    lang_dir = os.path.join(experiment_dir, language)
    
    if not os.path.exists(lang_dir):
        logger.warning(f"Language directory not found: {lang_dir}")
        return None
    
    # Load trained model
    logger.info(f"Loading model from {lang_dir}")
    try:
        model, tokenizer = load_trained_model_for_eval(
            model_config=model_config,
            label_info=label_info,
            checkpoint_dir=lang_dir,
            device=device
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None
    
    # Load data
    processed_dir = base_config['paths']['data_processed']
    data_path = os.path.join(processed_dir, f'{split}.csv')
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return None
    
    # Get max sequence length
    max_seq_len = base_config.get('data', {}).get('max_seq_len', 256)
    if 'model' in model_config and 'tokenizer' in model_config['model']:
        max_seq_len = model_config['model']['tokenizer'].get('max_length', max_seq_len)
    
    # Create dataset
    dataset = MultilingualClinicalDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        label2id=label_info['label2id'],
        max_length=max_seq_len,
        language=language
    )
    
    if len(dataset) == 0:
        logger.warning(f"No samples found for language {language} in {split} split")
        return None
    
    logger.info(f"Loaded {len(dataset)} samples for {language} on {split} split")
    
    # Create data loader
    batch_size = int(base_config.get('training', {}).get('batch_size_effective', 
                                                         base_config.get('training', {}).get('batch_size', 32)))
    data_collator = DataCollator(tokenizer, padding='longest')
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        collate_fn=data_collator,
        drop_last=False
    )
    
    # Run inference
    all_predictions = []
    all_labels = []
    all_logits = []
    total_loss = 0.0
    num_batches = 0
    
    model.eval()
    loss_fct = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Inference [{language}]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, labels=labels)
            
            # Get logits
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Compute loss
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            else:
                loss = loss_fct(logits, labels)
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(torch.softmax(logits, dim=-1).cpu().numpy())
            
            total_loss += loss.item()
            num_batches += 1
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    
    # Compute metrics
    metric_calc = MetricCalculator(label_info['id2label'])
    detailed_metrics = metric_calc.compute_detailed_metrics(
        all_predictions.tolist(),
        all_labels.tolist(),
        [language] * len(all_labels)
    )
    
    # Extract key metrics
    overall_metrics = detailed_metrics['overall']
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    results = {
        'model_name': model_config['model']['name'],
        'head_type': model_config['model'].get('head_type', 'linear'),
        'language': language,
        'split': split,
        'metrics': {
            'accuracy': overall_metrics.get('accuracy', 0.0),
            'f1_macro': overall_metrics.get('f1_macro', 0.0),
            'f1_weighted': overall_metrics.get('f1_weighted', 0.0),
            'precision_macro': overall_metrics.get('precision_macro', 0.0),
            'recall_macro': overall_metrics.get('recall_macro', 0.0),
            'loss': avg_loss
        },
        'n_samples': len(all_labels),
        'timestamp': datetime.now().isoformat()
    }
    
    # Print summary
    logger.info(f"[EVAL] Model={results['model_name']} | Lang={language} | Split={split} | "
                f"Acc={results['metrics']['accuracy']:.4f} | F1_macro={results['metrics']['f1_macro']:.4f}")
    
    return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained models on test/val/train splits',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to base configuration file (e.g., src/config/base_distilbert.yaml)'
    )
    parser.add_argument(
        '--experiment-id',
        type=str,
        default=None,
        help='Experiment ID (e.g., E1764404021). If not provided, uses most recent experiment.'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Data split to evaluate on (default: test)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    base_config = load_config(args.config)
    
    # Check for models section
    if 'models' not in base_config:
        raise ValueError(
            "Config must contain a 'models:' section. "
            "Please use one of the per-backbone config files."
        )
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load label info
    processed_dir = base_config['paths']['data_processed']
    labels_path = os.path.join(processed_dir, 'labels.json')
    label_info = load_label_mapping(labels_path)
    
    # Get experiment root and ID
    experiment_root = base_config.get('training', {}).get('experiment_root', 'experiments')
    if not os.path.isabs(experiment_root):
        if not experiment_root.startswith('experiments'):
            base_experiments = base_config['paths']['experiments']
            experiment_root = os.path.join(base_experiments, experiment_root)
    
    if args.experiment_id:
        experiment_id = args.experiment_id
    else:
        experiment_id = find_latest_experiment(experiment_root)
        if experiment_id is None:
            raise ValueError(
                f"No experiments found in {experiment_root}. "
                "Please train a model first or specify --experiment-id."
            )
        logger.info(f"Using latest experiment: {experiment_id}")
    
    experiment_dir = os.path.join(experiment_root, experiment_id)
    
    if not os.path.exists(experiment_dir):
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    
    logger.info(f"Evaluating experiment: {experiment_id}")
    logger.info(f"Experiment directory: {experiment_dir}")
    
    # Get languages from config
    languages = base_config.get('data', {}).get('languages', ['en', 'hi', 'pa'])
    models_cfg = base_config['models']
    
    # Evaluate each language
    all_results = {}
    
    for lang in languages:
        if lang not in models_cfg:
            logger.warning(f"Language '{lang}' not in models: section. Skipping.")
            continue
        
        # Build model config for this language
        model_config = build_model_config_from_models_section(
            models_cfg=models_cfg,
            language=lang,
            base_config=base_config
        )
        
        # Evaluate
        results = evaluate_language(
            language=lang,
            model_config=model_config,
            base_config=base_config,
            label_info=label_info,
            split=args.split,
            experiment_dir=experiment_dir,
            device=device
        )
        
        if results:
            all_results[lang] = results
            
            # Save per-language results
            lang_dir = os.path.join(experiment_dir, lang)
            os.makedirs(lang_dir, exist_ok=True)
            
            eval_file = os.path.join(lang_dir, f'eval_{args.split}.json')
            results['experiment_id'] = experiment_id
            
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved evaluation results to {eval_file}")
    
    # Print overall summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    for lang, results in all_results.items():
        metrics = results['metrics']
        logger.info(
            f"{lang.upper()}: Acc={metrics['accuracy']:.4f}, "
            f"F1_macro={metrics['f1_macro']:.4f}, "
            f"F1_weighted={metrics['f1_weighted']:.4f}"
        )
    logger.info("="*60)


if __name__ == '__main__':
    main()

