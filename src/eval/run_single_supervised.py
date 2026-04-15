"""
Evaluate a single trained model on test/val/train split.

Loads model from checkpoint directory and computes metrics.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import MultilingualClinicalDataset, DataCollator, load_label_mapping
from src.models.metrics import MetricCalculator
from src.evaluate.evaluate import load_model_from_checkpoint

logger = logging.getLogger(__name__)


def load_model_config_from_checkpoint(checkpoint_dir: str) -> Optional[Dict[str, Any]]:
    """
    Try to infer model config from checkpoint directory structure.
    
    Checks:
    1. Checkpoint file itself (if it contains config)
    2. Parent experiment directory summary.json
    3. Experiment root summary.json
    """
    # Try loading from checkpoint file
    checkpoint_file = os.path.join(checkpoint_dir, 'best.ckpt')
    if os.path.exists(checkpoint_file):
        try:
            import torch
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            if 'config' in checkpoint:
                return checkpoint['config']
        except Exception as e:
            logger.debug(f"Could not load config from checkpoint: {e}")
    
    # Try parent directory (experiment level)
    parent_dir = os.path.dirname(checkpoint_dir)
    summary_file = os.path.join(parent_dir, 'summary.json')
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
                if 'config' in summary:
                    return summary['config']
        except Exception as e:
            logger.debug(f"Could not load config from summary.json: {e}")
    
    # Try experiment root summary.json
    exp_root = os.path.dirname(parent_dir)
    summary_file = os.path.join(exp_root, 'summary.json')
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
                if 'config' in summary:
                    return summary['config']
        except Exception as e:
            logger.debug(f"Could not load config from root summary.json: {e}")
    
    return None


def evaluate_single_model(
    model_root: str,
    output_dir: str,
    split: str = 'test',
    device: Optional[torch.device] = None
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a single trained model.
    
    Args:
        model_root: Directory containing best.ckpt or model/ subdirectory
        output_dir: Directory to save evaluation results
        split: Data split ('train', 'val', or 'test')
        device: Device to use (default: auto-detect)
        
    Returns:
        Dictionary containing evaluation results, or None if evaluation failed
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Evaluating model from: {model_root}")
    logger.info(f"Split: {split}, Device: {device}")
    
    # Check for checkpoint
    checkpoint_path = os.path.join(model_root, 'best.ckpt')
    model_dir = os.path.join(model_root, 'model')
    
    if not os.path.exists(checkpoint_path) and not os.path.exists(model_dir):
        logger.error(f"No checkpoint found in {model_root}")
        return None
    
    # Try to load config from checkpoint or parent directory
    config = load_model_config_from_checkpoint(model_root)
    if config is None:
        logger.warning("Could not load model config. Using minimal config.")
        # We'll need to infer from the model itself
        config = {}
    
    # Load label info
    project_root = Path(__file__).resolve().parents[2]
    labels_path = project_root / 'data' / 'processed' / 'labels.json'
    if not labels_path.exists():
        logger.error(f"Labels file not found: {labels_path}")
        return None
    
    label_info = load_label_mapping(str(labels_path))
    
    # Try to infer model name from checkpoint or config
    model_name = 'unknown'
    head_type = 'linear'
    
    # Try to get model info from checkpoint
    if os.path.exists(checkpoint_path):
        try:
            import torch
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'config' in checkpoint and 'model' in checkpoint['config']:
                model_cfg = checkpoint['config']['model']
                model_name = model_cfg.get('name', model_name)
                head_type = model_cfg.get('head_type', head_type)
        except Exception:
            pass
    
    # Try to get from loaded config
    if config and 'model' in config:
        model_cfg = config['model']
        model_name = model_cfg.get('name', model_name)
        head_type = model_cfg.get('head_type', head_type)
    
    # Try to infer from model directory (for saved transformers models)
    if os.path.exists(model_dir):
        try:
            from transformers import AutoConfig
            model_config_hf = AutoConfig.from_pretrained(model_dir)
            # Try to get model name from config
            if hasattr(model_config_hf, '_name_or_path'):
                model_name = model_config_hf._name_or_path
        except Exception:
            pass
    
    # Build model config for loading
    model_config = {
        'model': {
            'name': model_name,
            'tokenizer': {'name': model_name, 'max_length': max_length},
            'head_type': head_type
        }
    }
    
    # Add HPA config if needed
    if head_type == 'hpa' and config and 'model' in config:
        if 'head' in config['model']:
            model_config['model']['head'] = config['model']['head']
        if 'backbone' in config['model']:
            model_config['model']['backbone'] = config['model']['backbone']
    
    # Try to load model
    try:
        model, tokenizer = load_model_from_checkpoint(
            checkpoint_path=checkpoint_path if os.path.exists(checkpoint_path) else model_dir,
            model_config=model_config,
            label_info=label_info,
            device=device
        )
        
        model.eval()
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    # Load test data
    data_path = project_root / 'data' / 'processed' / f'{split}.csv'
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return None
    
    # Infer language from model_root path
    language = os.path.basename(model_root)
    if language not in ['en', 'hi', 'pa']:
        # Try parent directory
        parent = os.path.basename(os.path.dirname(model_root))
        if parent in ['en', 'hi', 'pa']:
            language = parent
        else:
            logger.warning(f"Could not infer language from path. Using 'en' as default.")
            language = 'en'
    
    # Get max length from config or default
    max_length = 256
    if config and 'data' in config:
        max_length = config['data'].get('max_seq_len', 256)
    
    # Create dataset
    dataset = MultilingualClinicalDataset(
        data_path=str(data_path),
        tokenizer=tokenizer,
        label2id=label_info['label2id'],
        max_length=max_length,
        language=language
    )
    
    if len(dataset) == 0:
        logger.warning(f"No samples found for language {language} in {split} split")
        return None
    
    logger.info(f"Loaded {len(dataset)} samples for {language} on {split} split")
    
    # Create data loader
    batch_size = 32
    if config and 'training' in config:
        batch_size = config['training'].get('batch_size_effective', 
                                           config['training'].get('batch_size', 32))
    
    data_collator = DataCollator(tokenizer, padding='longest')
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=data_collator,
        drop_last=False
    )
    
    # Run inference
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
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
            
            total_loss += loss.item()
            num_batches += 1
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
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
        'model_root': model_root,
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
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved metrics to {metrics_file}")
    logger.info(
        f"Results: Acc={results['metrics']['accuracy']:.4f}, "
        f"F1_macro={results['metrics']['f1_macro']:.4f}"
    )
    
    return results


def main():
    """Main function for single model evaluation."""
    parser = argparse.ArgumentParser(
        description='Evaluate a single trained model on test/val/train split',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--model_root',
        type=str,
        required=True,
        help='Directory containing best.ckpt or model/ subdirectory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save evaluation results'
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
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Evaluate
    results = evaluate_single_model(
        model_root=args.model_root,
        output_dir=args.output_dir,
        split=args.split,
        device=device
    )
    
    if results is None:
        raise SystemExit("Evaluation failed")


if __name__ == '__main__':
    main()

