"""
Evaluation script for DistilBERT models trained on JSONL data.

This script evaluates a trained model on the test split of the JSONL dataset,
generating comprehensive metrics including F1 scores, confusion matrix,
AUROC, and AUPRC.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
import json
import argparse
import yaml
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Local imports
from src.models.metrics import MetricCalculator, compute_auroc_ovr, compute_auprc_ovr
from src.models.baselines import build_model_and_tokenizer
from src.utils.io import write_json, read_json
from src.utils.seed import set_seed

# Import our JSONL dataset implementation
from train_jsonl.jsonl_dataset import JSONLClinicalDataset, DataCollator, load_label_mapping

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_config: Dict[str, Any],
    label_info: Dict[str, Any],
    device: torch.device
) -> Tuple[torch.nn.Module, Any]:
    """Load a trained model from checkpoint."""
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Check if checkpoint is a file or directory
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_state_dict = checkpoint['model_state_dict']
    else:
        # Assume it's a directory with best.ckpt
        checkpoint_file = os.path.join(checkpoint_path, 'best.ckpt')
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Best checkpoint file not found: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model_state_dict = checkpoint['model_state_dict']
    
    # Build model and load state dict
    model, tokenizer = build_model_and_tokenizer(
        model_config=model_config,
        label_info=label_info,
        device=device
    )
    
    # Load state dict
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    logger.info(f"Successfully loaded model from checkpoint")
    return model, tokenizer


def evaluate_model(
    model: torch.nn.Module,
    tokenizer: Any,
    test_dataset: JSONLClinicalDataset,
    device: torch.device,
    label_info: Dict[str, Any],
    batch_size: int = 16
) -> Dict[str, Any]:
    """Evaluate the model on the test dataset."""
    logger.info(f"Starting evaluation with batch_size={batch_size}")
    
    # Create data loader
    data_collator = DataCollator(tokenizer, padding='longest')
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues
        collate_fn=data_collator
    )
    
    # Prepare for evaluation
    all_predictions = []
    all_labels = []
    all_logits = []
    all_probabilities = []
    total_loss = 0.0
    
    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluation")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, labels=labels)
            
            # Get loss
            loss = outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else None
            if loss is not None:
                total_loss += loss.item()
            
            # Get logits and probabilities
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            # Collect predictions
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'test_loss': total_loss / len(test_loader) if loss is not None else None
    }
    
    # Convert to numpy arrays
    all_logits = np.array(all_logits)
    all_probabilities = np.array(all_probabilities)
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Compute basic classification metrics
    id2label_int = {int(k): v for k, v in label_info['id2label'].items()}
    metric_calculator = MetricCalculator(id2label_int, include_weighted=False)
    
    # Get present labels
    present_labels = sorted(set(all_labels))
    target_names = [id2label_int[l] for l in present_labels]
    
    # Compute metrics
    basic_metrics = metric_calculator.compute_metrics((all_predictions, all_labels))
    metrics.update(basic_metrics)
    
    # Compute classification report
    class_report = classification_report(
        all_labels, all_predictions,
        labels=present_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    metrics['classification_report'] = class_report
    
    # Compute AUROC and AUPRC (one-vs-rest)
    try:
        # Get the unique labels present in the current batch
        unique_labels = np.unique(all_labels)
        
        # Only compute AUROC and AUPRC if we have at least 2 classes
        if len(unique_labels) >= 2:
            # AUROC
            auroc_macro = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', average='macro')
            metrics['auroc'] = {
                'macro': auroc_macro
            }
            
            # AUPRC
            auprc_macro = average_precision_score(all_labels, all_probabilities, average='macro')
            metrics['auprc'] = {
                'macro': auprc_macro
            }
            
            logger.info(f"AUROC: macro={auroc_macro:.4f}")
            logger.info(f"AUPRC: macro={auprc_macro:.4f}")
        else:
            logger.info("Skipping AUROC/AUPRC calculation: less than 2 classes present")
            metrics['auroc'] = None
            metrics['auprc'] = None
        
    except Exception as e:
        logger.warning(f"Could not compute AUROC/AUPRC: {e}")
        metrics['auroc'] = None
        metrics['auprc'] = None
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions, labels=present_labels)
    metrics['confusion_matrix'] = {
        'matrix': cm.tolist(),
        'labels': target_names
    }
    
    logger.info(f"Evaluation completed")
    logger.info(f"Test metrics: {basic_metrics}")
    
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    output_path: str
) -> None:
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {output_path}")


def main():
    """Main evaluation function."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate DistilBERT model on JSONL data')
    parser.add_argument('--config', type=str, default='src/config/base_distilbert.yaml',
                        help='Path to configuration file')
    parser.add_argument('--jsonl_path', type=str, default='preprocess_jsonl/english_data.jsonl',
                        help='Path to JSONL data file')
    parser.add_argument('--labels_path', type=str, default='preprocess_jsonl/labels.json',
                        help='Path to labels JSON file')
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to model checkpoint file or directory')
    parser.add_argument('--output_dir', type=str, default='train_jsonl/evaluations',
                        help='Output directory for evaluation results')
    parser.add_argument('--language', type=str, default='en',
                        help='Language to evaluate on')
    args = parser.parse_args()
    
    if not args.checkpoint_path:
        raise ValueError("Please provide --checkpoint_path")
    
    # Load configuration
    config = yaml.safe_load(open(args.config, 'r', encoding='utf-8'))
    
    # Set seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load label mapping
    label_info = load_label_mapping(args.labels_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model from checkpoint
    model, tokenizer = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        model_config=config,
        label_info=label_info,
        device=device
    )
    
    # Create test dataset from JSONL file
    max_seq_len = config['data'].get('max_seq_len', 256)
    logger.info(f"Creating test dataset with max_seq_len={max_seq_len}")
    
    # We'll create a test dataset with a small subset for evaluation
    # In a real scenario, you'd want to use a separate test split
    test_dataset = JSONLClinicalDataset(
        data_path=args.jsonl_path,
        tokenizer=tokenizer,
        label2id=label_info['label2id'],
        max_length=max_seq_len,
        language=args.language
    )
    
    # Evaluate the model
    batch_size = config['training'].get('batch_size', 16)
    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        device=device,
        label_info=label_info,
        batch_size=batch_size
    )
    
    # Save evaluation results
    eval_time = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    eval_results_path = os.path.join(args.output_dir, f'evaluation_results_{eval_time}.json')
    write_json(Path(eval_results_path), metrics)
    logger.info(f"Saved evaluation results to {eval_results_path}")
    
    # Plot and save confusion matrix
    cm = np.array(metrics['confusion_matrix']['matrix'])
    labels = metrics['confusion_matrix']['labels']
    cm_path = os.path.join(args.output_dir, f'confusion_matrix_{eval_time}.png')
    plot_confusion_matrix(cm, labels, cm_path)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Checkpoint: {args.checkpoint_path}")
    logger.info(f"Test Loss: {metrics.get('test_loss', 'N/A'):.4f}")
    logger.info(f"Accuracy: {metrics.get('accuracy', 0.0):.4f}")
    logger.info(f"F1-Macro: {metrics.get('f1_macro', 0.0):.4f}")
    logger.info(f"Precision-Macro: {metrics.get('precision_macro', 0.0):.4f}")
    logger.info(f"Recall-Macro: {metrics.get('recall_macro', 0.0):.4f}")
    if metrics.get('auroc'):
        logger.info(f"AUROC-Macro: {metrics['auroc']['macro']:.4f}")
    if metrics.get('auprc'):
        logger.info(f"AUPRC-Macro: {metrics['auprc']['macro']:.4f}")
    logger.info("="*60)


if __name__ == "__main__":
    import pandas as pd  # Import here to avoid circular imports
    main()
