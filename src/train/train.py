"""
Training script for multilingual clinical NLP models.

Implements the methodology:
- English → DistilBERT + linear head
- Hindi & Punjabi → IndicBERT-HPA
- Supervised multi-class diagnosis classification
- AdamW + warmup + discriminative LRs + early stopping
"""  ""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
import shutil
import time
import json
import argparse
import yaml
from tqdm import tqdm
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.metrics import MetricCalculator, compute_metrics_from_logits
from src.models.baselines import build_model_and_tokenizer
from src.data.dataset import (
    MultilingualClinicalDataset, 
    DataCollator, 
    load_label_mapping,
    create_data_loaders
)
from src.utils.io import write_json, read_json
from pathlib import Path
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def calculate_class_weights_inverse_frequency(
    labels: List[int], 
    num_classes: int,
    smoothing: float = 0.1
) -> torch.Tensor:
    """
    Calculate class weights using inverse frequency with smoothing.
    
    Args:
        labels: List of label IDs
        num_classes: Total number of classes
        smoothing: Smoothing factor to prevent extreme weights
    
    Returns:
        Tensor of class weights
    """
    # Count class frequencies
    class_counts = Counter(labels)
    
    # Calculate inverse frequencies
    total_samples = len(labels)
    weights = []
    
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 0)
        if count == 0:
            # If class not present, use average weight
            freq = 1.0 / num_classes
        else:
            freq = count / total_samples
        
        # Inverse frequency with smoothing
        inv_freq = 1.0 / (freq + smoothing)
        weights.append(inv_freq)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum() * num_classes  # Normalize to have mean = 1
    
    return torch.tensor(weights, dtype=torch.float32)


def label_smoothing_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    smoothing: float = 0.05,
    class_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute cross-entropy loss with label smoothing.
    
    Args:
        logits: Model logits [batch_size, num_classes]
        labels: Ground truth labels [batch_size]
        num_classes: Number of classes
        smoothing: Label smoothing factor (ε ≈ 0.05)
        class_weights: Optional class weights tensor
    
    Returns:
        Loss tensor
    """
    # Convert labels to one-hot
    batch_size = labels.size(0)
    true_dist = torch.zeros_like(logits)
    true_dist.scatter_(1, labels.unsqueeze(1), 1.0)
    
    # Apply label smoothing
    smooth_dist = true_dist * (1.0 - smoothing) + smoothing / num_classes
    
    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    
    # Compute loss
    loss = -torch.sum(smooth_dist * log_probs, dim=1)
    
    # Apply class weights if provided
    if class_weights is not None:
        class_weights = class_weights.to(loss.device)
        weights = class_weights[labels]
        loss = loss * weights
    
    return loss.mean()


def get_optimizer_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float = 0.01,
    no_decay: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Get parameter groups for optimizer with proper weight decay exclusions.
    
    If model has get_param_groups method (e.g., IndicBERT-HPA), use it for discriminative LRs.
    Otherwise, create standard groups with weight decay exclusions.
    
    Args:
        model: PyTorch model
        base_lr: Base learning rate (must be float)
        weight_decay: Weight decay coefficient (must be float)
        no_decay: List of parameter names to exclude from weight decay
    
    Returns:
        List of parameter group dictionaries with all 'lr' values as floats
    """
    # Ensure base_lr and weight_decay are floats
    base_lr = float(base_lr)
    weight_decay = float(weight_decay)
    
    no_decay = no_decay or ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    
    # Check if model has get_param_groups method (for discriminative LRs)
    if hasattr(model, 'get_param_groups'):
        logger.info("Using model's get_param_groups for discriminative learning rates")
        param_groups = model.get_param_groups(base_lr)
        
        # Ensure all learning rates in groups are floats
        for group in param_groups:
            if 'lr' in group:
                group['lr'] = float(group['lr'])
            if 'weight_decay' not in group:
                group['weight_decay'] = float(weight_decay)
            elif 'weight_decay' in group:
                group['weight_decay'] = float(group['weight_decay'])
        
        return param_groups
    
    # Standard parameter grouping with weight decay exclusions
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'lr': float(base_lr), 'weight_decay': float(weight_decay)},
        {'params': no_decay_params, 'lr': float(base_lr), 'weight_decay': 0.0}
    ]
    
    return param_groups


class Trainer:
    """Trainer for multilingual clinical NLP models with language-specific training."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        experiment_dir: str,
        label_info: Dict[str, Any],
        config: Dict[str, Any],
        language: str,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.experiment_dir = experiment_dir
        self.label_info = label_info
        self.config = config
        self.language = language
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.best_val_f1_macro = 0.0
        self.best_val_f1_weighted = 0.0
        self.best_val_balanced_accuracy = 0.0
        self.best_epoch = 0
        self.best_epoch_f1_macro = 0
        self.best_epoch_f1_weighted = 0
        self.best_epoch_balanced_accuracy = 0
        self.patience_counter = 0
        self.epochs_since_improvement = 0  # Track epochs since last improvement
        
        # Metrics calculator (convert id2label keys to int)
        id2label_int = {int(k): v for k, v in label_info['id2label'].items()}
        self.metric_calculator = MetricCalculator(id2label_int)
        self.label_info = label_info  # Store for confusion matrix
        
        # Ensure experiment directory exists
        lang_dir = os.path.join(experiment_dir, language)
        os.makedirs(lang_dir, exist_ok=True)
        os.makedirs(os.path.join(lang_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(lang_dir, 'best_models'), exist_ok=True)
        
        logger.info(f"Trainer initialized for language: {language}, device: {device}")
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss with optional label smoothing and class weights."""
        if self.label_smoothing > 0:
            num_classes = logits.size(-1)
            return label_smoothing_loss(
                logits, labels, num_classes, 
                self.label_smoothing, self.class_weights
            )
        else:
            # Standard cross-entropy
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            return loss_fct(logits, labels)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1} [{self.language}]")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, labels=labels)
            
            # Compute loss (use model's loss if available, otherwise compute manually)
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            else:
                loss = self.compute_loss(outputs.logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            gradient_clip_norm = float(self.config['training'].get('gradient_clip_norm', 1.0))
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=gradient_clip_norm)
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            self.global_step += 1
        
        # Compute epoch metrics
        epoch_metrics = self.metric_calculator.compute_metrics(
            (np.array(all_predictions), np.array(all_labels))
        )
        epoch_metrics['loss'] = total_loss / len(self.train_loader)
        
        # Make training metrics logging more explicit
        logger.info(f"Epoch {self.epoch + 1} [{self.language}] training metrics:")
        logger.info(f"  Loss: {epoch_metrics['loss']:.4f}")
        logger.info(f"  Accuracy: {epoch_metrics['accuracy']:.4f} (regular accuracy)")
        logger.info(f"  Balanced Accuracy: {epoch_metrics['balanced_accuracy']:.4f} (per-class average)")
        logger.info(f"  F1 Macro: {epoch_metrics['f1_macro']:.4f}")
        logger.info(f"  F1 Weighted: {epoch_metrics['f1_weighted']:.4f}")
        if 'auroc_macro' in epoch_metrics:
            logger.info(f"  AUROC Macro: {epoch_metrics['auroc_macro']:.4f}")
        if 'auprc_macro' in epoch_metrics:
            logger.info(f"  AUPRC Macro: {epoch_metrics['auprc_macro']:.4f}")
        if 'ece' in epoch_metrics:
            logger.info(f"  ECE: {epoch_metrics['ece']:.4f}")
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, Any]:
        """Validate model on validation set."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_logits = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validation [{self.language}]"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, labels=labels)
                
                # Compute loss
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss = outputs.loss
                else:
                    loss = self.compute_loss(outputs.logits, labels)
                
                # Collect predictions and logits
                total_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(torch.softmax(outputs.logits, dim=-1).cpu().numpy())
        
        # Log prediction distribution (for majority class detection)
        pred_counts = Counter(all_predictions)
        total_preds = len(all_predictions)
        logger.info(f"\n[{self.language}] Prediction distribution on validation:")
        for pred_id, count in pred_counts.most_common():
            pred_label = self.label_info['id2label'].get(pred_id, f"ID_{pred_id}")
            ratio = count / total_preds
            logger.info(f"  {pred_label} (ID {pred_id}): {count}/{total_preds} = {ratio:.2%}")
        
        # Check for majority class collapse (>80% same prediction)
        max_pred_ratio = max(pred_counts.values()) / total_preds if total_preds > 0 else 0
        if max_pred_ratio > 0.8:
            majority_pred_id = pred_counts.most_common(1)[0][0]
            majority_label = self.label_info['id2label'].get(majority_pred_id, f"ID_{majority_pred_id}")
            logger.warning(f"⚠️  MAJORITY CLASS COLLAPSE in predictions!")
            logger.warning(f"   {majority_label} (ID {majority_pred_id}) represents {max_pred_ratio:.2%} of predictions")
            logger.warning(f"   Model is likely predicting only the majority class")
        
        # Compute detailed metrics
        val_metrics = self.metric_calculator.compute_detailed_metrics(
            all_predictions, all_labels, [self.language] * len(all_labels)
        )
        val_metrics['overall']['loss'] = total_loss / len(self.val_loader)
        
        # Add per-class F1 to metrics
        present_label_ids = sorted(set(all_labels))
        target_names = [self.label_info['id2label'].get(int(i), f"ID_{i}") for i in present_label_ids]
        class_report = classification_report(
            all_labels, all_predictions,
            labels=present_label_ids,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        val_metrics['per_class'] = class_report
        
        # Compute AUROC and AUPRC (one-vs-rest) from logits/probabilities
        from src.models.metrics import compute_auroc_ovr, compute_auprc_ovr
        try:
            all_logits_array = np.array(all_logits)
            all_labels_array = np.array(all_labels)
            
            # Compute AUROC
            auroc_results = compute_auroc_ovr(
                all_labels_array, all_logits_array, 
                labels=present_label_ids
            )
            val_metrics['auroc'] = auroc_results
            
            # Compute AUPRC
            auprc_results = compute_auprc_ovr(
                all_labels_array, all_logits_array,
                labels=present_label_ids
            )
            val_metrics['auprc'] = auprc_results
            
            # Add to overall metrics for easy access
            val_metrics['overall']['auroc_macro'] = auroc_results.get('macro', 0.0)
            val_metrics['overall']['auroc_micro'] = auroc_results.get('micro', 0.0)
            val_metrics['overall']['auprc_macro'] = auprc_results.get('macro', 0.0)
            val_metrics['overall']['auprc_micro'] = auprc_results.get('micro', 0.0)
        except Exception as e:
            logger.warning(f"Could not compute AUROC/AUPRC: {e}")
            val_metrics['auroc'] = {'macro': None, 'micro': None}
            val_metrics['auprc'] = {'macro': None, 'micro': None}
        
        # Make metrics logging more explicit to distinguish between regular and balanced accuracy
        logger.info(f"Validation metrics [{self.language}]:")
        logger.info(f"  Loss: {val_metrics['overall']['loss']:.4f}")
        logger.info(f"  Accuracy: {val_metrics['overall']['accuracy']:.4f} (regular accuracy)")
        logger.info(f"  Balanced Accuracy: {val_metrics['overall']['balanced_accuracy']:.4f} (per-class average)")
        logger.info(f"  F1 Macro: {val_metrics['overall']['f1_macro']:.4f}")
        logger.info(f"  F1 Weighted: {val_metrics['overall']['f1_weighted']:.4f}")
        if 'auroc_macro' in val_metrics['overall']:
            logger.info(f"  AUROC Macro: {val_metrics['overall']['auroc_macro']:.4f}")
        if 'auprc_macro' in val_metrics['overall']:
            logger.info(f"  AUPRC Macro: {val_metrics['overall']['auprc_macro']:.4f}")
        if 'ece' in val_metrics['overall']:
            logger.info(f"  ECE: {val_metrics['overall']['ece']:.4f}")
        
        # Save confusion matrix and classification report
        self.save_confusion_matrix_and_report(
            all_labels, all_predictions, 
            split_name='val',
            epoch=self.epoch
        )
        
        return val_metrics
    
    def save_confusion_matrix_and_report(
        self,
        y_true: List[int],
        y_pred: List[int],
        split_name: str = 'val',
        epoch: int = 0
    ):
        """Save confusion matrix plot and classification report JSON."""
        lang_dir = os.path.join(self.experiment_dir, self.language)
        
        # Get present labels
        present_labels = sorted(set(y_true + y_pred))
        if len(present_labels) == 0:
            logger.warning(f"No labels present for confusion matrix [{self.language}]")
            return
        
        # Get label names
        label_names = [self.label_info['id2label'].get(int(l), f"ID_{l}") for l in present_labels]
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=present_labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=label_names,
            yticklabels=label_names
        )
        plt.title(f'Confusion Matrix - {self.language} - {split_name} (Epoch {epoch})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_path = os.path.join(lang_dir, f'confusion_matrix_{split_name}_epoch{epoch}.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confusion matrix to {cm_path}")
        
        # Compute and save classification report
        class_report = classification_report(
            y_true, y_pred,
            labels=present_labels,
            target_names=label_names,
            output_dict=True,
            zero_division=0
        )
        
        report_path = os.path.join(lang_dir, f'classification_report_{split_name}_epoch{epoch}.json')
        write_json(Path(report_path), class_report)
        logger.info(f"Saved classification report to {report_path}")
        
        # Also save best epoch versions (overwrite on best)
        if split_name == 'val':
            cm_best_path = os.path.join(lang_dir, f'confusion_matrix_{split_name}_best.png')
            report_best_path = os.path.join(lang_dir, f'classification_report_{split_name}_best.json')
            try:
                shutil.copy(cm_path, cm_best_path)
                write_json(Path(report_best_path), class_report)
            except Exception as e:
                logger.warning(f"Could not save best confusion matrix: {e}")
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint with atomic writes and error handling.
        
        Uses temp file + rename pattern to avoid partial writes on Windows.
        Saves only state_dict() to reduce file size and avoid serialization issues.
        """
        # Get checkpoint directory (can be overridden via config)
        checkpoint_dir = self.config.get('training', {}).get('checkpoint_dir')
        if checkpoint_dir:
            # Use custom checkpoint directory, but maintain experiment structure
            experiment_root = checkpoint_dir
            experiment_id = os.path.basename(self.experiment_dir)
            lang_dir = os.path.join(experiment_root, experiment_id, self.language)
        else:
            # Use default experiment directory
            lang_dir = os.path.join(self.experiment_dir, self.language)
        
        # Ensure all directories exist
        os.makedirs(lang_dir, exist_ok=True)
        checkpoints_dir = os.path.join(lang_dir, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),  # Only save state_dict, not full model
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_f1_macro': self.best_val_f1_macro,
            'config': self.config,
            'language': self.language
        }
        
        def save_atomically(data: Dict, final_path: str, description: str = "checkpoint"):
            """
            Save checkpoint atomically: write to temp file, then rename.
            This avoids partial writes on Windows (Defender/antivirus locking).
            
            Uses os.replace() which is atomic on Windows when both files are on the same filesystem.
            """
            temp_path = None
            try:
                # Create temp file in same directory (required for atomic rename on Windows)
                temp_path = final_path + '.tmp'
                
                # Clean up any existing temp file from previous failed save
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as cleanup_e:
                        logger.warning(f"Could not clean up existing temp file {temp_path}: {cleanup_e}")
                
                # Ensure parent directory exists
                parent_dir = os.path.dirname(final_path)
                os.makedirs(parent_dir, exist_ok=True)
                
                # Write to temp file
                torch.save(data, temp_path)
                
                # Atomic replace (works on Windows if files are on same filesystem)
                # os.replace() is atomic on both Windows and Unix
                os.replace(temp_path, final_path)
                
                logger.debug(f"Successfully saved {description} to {final_path}")
                return True
                
            except Exception as e:
                # Clean up temp file if it exists
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
                logger.error(f"Failed to save {description} to {final_path}: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                # Check for common issues
                if "disk" in str(e).lower() or "space" in str(e).lower():
                    logger.error("Possible disk space issue - check available disk space")
                elif "permission" in str(e).lower() or "access" in str(e).lower():
                    logger.error("Possible permission issue - check file/directory permissions")
                elif "lock" in str(e).lower():
                    logger.error("Possible file lock issue - antivirus/Defender may be scanning the file")
                return False
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            checkpoints_dir, 
            f'checkpoint_epoch_{self.epoch}.pt'
        )
        
        if not save_atomically(checkpoint, checkpoint_path, f"checkpoint epoch {self.epoch}"):
            logger.warning(f"Failed to save checkpoint for epoch {self.epoch}, continuing training...")
        
        # Save best model
        if is_best:
            best_path = os.path.join(lang_dir, 'best.ckpt')
            
            if save_atomically(checkpoint, best_path, "best model"):
                logger.info(f"Saved best model for {self.language} to {best_path}")
            else:
                logger.warning(f"Failed to save best model for {self.language}, continuing training...")
            
            # Also save in transformers format if model supports it (with error handling)
            if hasattr(self.model, 'save_pretrained'):
                model_save_path = os.path.join(lang_dir, 'model')
                try:
                    os.makedirs(model_save_path, exist_ok=True)
                    self.model.save_pretrained(model_save_path)
                    logger.debug(f"Saved transformers model to {model_save_path}")
                except Exception as e:
                    logger.warning(f"Failed to save transformers model to {model_save_path}: {str(e)}")
                    logger.warning("Continuing training without transformers format save...")
    
    def save_best_checkpoint_by_metric(
        self, 
        metric_name: str, 
        epoch: int, 
        val_metrics: Dict[str, Any]
    ):
        """
        Save best checkpoint for a specific metric (f1_macro, f1_weighted, balanced_accuracy, accuracy).
        
        Saves to best_models/ folder with metric-specific naming.
        """
        lang_dir = os.path.join(self.experiment_dir, self.language)
        best_models_dir = os.path.join(lang_dir, 'best_models')
        os.makedirs(best_models_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metric_name': metric_name,
            'metric_value': val_metrics['overall'].get(metric_name, 0.0),
            'all_metrics': val_metrics['overall'],
            'config': self.config,
            'language': self.language
        }
        
        # Save checkpoint
        checkpoint_path = os.path.join(best_models_dir, f'best_{metric_name}.pt')
        temp_path = checkpoint_path + '.tmp'
        
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, checkpoint_path)
            logger.info(f"Saved best {metric_name} checkpoint to {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to save best {metric_name} checkpoint: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
        
        # Save confusion matrix and classification report for this best metric
        try:
            # Get predictions and labels from validation (we need to recompute or store them)
            # For now, we'll save the confusion matrix from the last validation
            # In a full implementation, we might want to store predictions/labels per epoch
            # For now, we'll copy the best confusion matrix if it exists
            cm_path = os.path.join(lang_dir, f'confusion_matrix_val_epoch{epoch}.png')
            cm_best_path = os.path.join(best_models_dir, f'confusion_matrix_best_{metric_name}.png')
            if os.path.exists(cm_path):
                shutil.copy(cm_path, cm_best_path)
            
            report_path = os.path.join(lang_dir, f'classification_report_val_epoch{epoch}.json')
            report_best_path = os.path.join(best_models_dir, f'classification_report_best_{metric_name}.json')
            if os.path.exists(report_path):
                shutil.copy(report_path, report_best_path)
        except Exception as e:
            logger.warning(f"Could not save confusion matrix/report for best {metric_name}: {e}")
    
    def train(self, epochs: int, early_stopping_patience: int = None) -> Dict[str, Any]:
        """Full training loop with early stopping."""
        training_history = {
            'train_metrics': [],
            'val_metrics': [],
            'best_epoch': 0,
            'config': self.config,
            'language': self.language
        }
        
        early_stopping_config = self.config['training'].get('early_stopping', {})
        early_stopping_patience = early_stopping_patience or int(early_stopping_config.get('patience', 
                                                                                          early_stopping_config.get('early_stopping_patience', 3)))
        min_delta = float(early_stopping_config.get('min_delta', 
                                                   early_stopping_config.get('early_stopping_min_delta', 0.0)))
        monitor_metric = early_stopping_config.get('monitor', 'val_loss').lower()
        
        logger.info(f"Starting training for {epochs} epochs [{self.language}]")
        logger.info(f"Early stopping: patience={early_stopping_patience}, min_delta={min_delta}, monitor={monitor_metric}")
        start_time = time.time()
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch()
            training_history['train_metrics'].append(train_metrics)
            
            # Validation phase
            val_metrics = self.validate()
            training_history['val_metrics'].append(val_metrics)
            
            current_val_loss = val_metrics['overall']['loss']
            current_val_accuracy = val_metrics['overall']['accuracy']
            current_val_f1_macro = val_metrics['overall'].get('f1_macro', 0.0)
            current_val_f1_weighted = val_metrics['overall'].get('f1_weighted', 0.0)
            current_val_balanced_accuracy = val_metrics['overall'].get('balanced_accuracy', 0.0)
            
            # Track best metrics independently (for multiple best checkpoints)
            is_best_f1_macro = False
            is_best_f1_weighted = False
            is_best_balanced_accuracy = False
            is_best_accuracy = False
            
            # Check for improvement in F1-macro
            if current_val_f1_macro > self.best_val_f1_macro + min_delta:
                self.best_val_f1_macro = current_val_f1_macro
                self.best_epoch_f1_macro = epoch
                is_best_f1_macro = True
                logger.info(f"✓ New best F1-macro! [{self.language}] F1-macro: {current_val_f1_macro:.4f} (epoch {epoch + 1})")
            
            # Check for improvement in F1-weighted
            if current_val_f1_weighted > self.best_val_f1_weighted + min_delta:
                self.best_val_f1_weighted = current_val_f1_weighted
                self.best_epoch_f1_weighted = epoch
                is_best_f1_weighted = True
                logger.info(f"✓ New best F1-weighted! [{self.language}] F1-weighted: {current_val_f1_weighted:.4f} (epoch {epoch + 1})")
            
            # Check for improvement in balanced accuracy
            if current_val_balanced_accuracy > self.best_val_balanced_accuracy + min_delta:
                self.best_val_balanced_accuracy = current_val_balanced_accuracy
                self.best_epoch_balanced_accuracy = epoch
                is_best_balanced_accuracy = True
                logger.info(f"✓ New best balanced accuracy! [{self.language}] Balanced acc: {current_val_balanced_accuracy:.4f} (epoch {epoch + 1})")
            
            # Check for improvement in accuracy
            if current_val_accuracy > self.best_val_accuracy + min_delta:
                self.best_val_accuracy = current_val_accuracy
                is_best_accuracy = True
            
            # Determine which metric to monitor for early stopping (primary monitor)
            is_best = False
            improved = False
            
            if monitor_metric == 'val_loss':
                # For loss, lower is better
                improvement = self.best_val_loss - current_val_loss
                if improvement > min_delta:
                    self.best_val_loss = current_val_loss
                    self.best_val_accuracy = current_val_accuracy
                    self.best_val_f1_macro = current_val_f1_macro
                    self.best_epoch = epoch
                    self.epochs_since_improvement = 0
                    improved = True
                    is_best = True
                    logger.info(f"✓ Improvement! [{self.language}] Loss: {current_val_loss:.4f} (improved by {improvement:.4f}), "
                              f"Accuracy: {current_val_accuracy:.4f}, F1: {current_val_f1_macro:.4f}")
                else:
                    self.epochs_since_improvement += 1
                    
            elif monitor_metric == 'val_accuracy':
                # For accuracy, higher is better
                improvement = current_val_accuracy - self.best_val_accuracy
                if improvement > min_delta:
                    self.best_val_loss = current_val_loss
                    self.best_val_accuracy = current_val_accuracy
                    self.best_val_f1_macro = current_val_f1_macro
                    self.best_epoch = epoch
                    self.epochs_since_improvement = 0
                    improved = True
                    is_best = True
                    logger.info(f"✓ Improvement! [{self.language}] Accuracy: {current_val_accuracy:.4f} (improved by {improvement:.4f}), "
                              f"Loss: {current_val_loss:.4f}, F1: {current_val_f1_macro:.4f}")
                else:
                    self.epochs_since_improvement += 1
                    
            elif monitor_metric == 'val_f1_macro':
                # For F1, higher is better
                improvement = current_val_f1_macro - self.best_val_f1_macro
                if improvement > min_delta:
                    self.best_val_loss = current_val_loss
                    self.best_val_accuracy = current_val_accuracy
                    self.best_val_f1_macro = current_val_f1_macro
                    self.best_epoch = epoch
                    self.epochs_since_improvement = 0
                    improved = True
                    is_best = True
                    logger.info(f"✓ Improvement! [{self.language}] F1: {current_val_f1_macro:.4f} (improved by {improvement:.4f}), "
                              f"Loss: {current_val_loss:.4f}, Accuracy: {current_val_accuracy:.4f}")
                else:
                    self.epochs_since_improvement += 1
            else:
                # Default to val_loss
                improvement = self.best_val_loss - current_val_loss
                if improvement > min_delta:
                    self.best_val_loss = current_val_loss
                    self.best_val_accuracy = current_val_accuracy
                    self.best_val_f1_macro = current_val_f1_macro
                    self.best_epoch = epoch
                    self.epochs_since_improvement = 0
                    improved = True
                    is_best = True
                    logger.info(f"✓ Improvement! [{self.language}] Loss: {current_val_loss:.4f} (improved by {improvement:.4f})")
                else:
                    self.epochs_since_improvement += 1
            
            if not improved:
                logger.info(f"  No improvement [{self.language}] Epoch {epoch + 1}/{epochs} - "
                          f"Epochs since improvement: {self.epochs_since_improvement}/{early_stopping_patience}")
            
            # Save checkpoints (with error handling - won't crash training if save fails)
            try:
                # Save primary best checkpoint (based on monitor metric)
                self.save_checkpoint(is_best=is_best)
                
                # Save multiple best checkpoints by different metrics
                if is_best_f1_macro:
                    self.save_best_checkpoint_by_metric('f1_macro', epoch, val_metrics)
                if is_best_f1_weighted:
                    self.save_best_checkpoint_by_metric('f1_weighted', epoch, val_metrics)
                if is_best_balanced_accuracy:
                    self.save_best_checkpoint_by_metric('balanced_accuracy', epoch, val_metrics)
                if is_best_accuracy:
                    self.save_best_checkpoint_by_metric('accuracy', epoch, val_metrics)
            except Exception as e:
                logger.error(f"Unexpected error during checkpoint save: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.warning("Continuing training despite checkpoint save failure...")
            
            # Early stopping check
            if self.epochs_since_improvement >= early_stopping_patience:
                logger.info(f"⏹ Early stopping triggered after {epoch + 1} epochs [{self.language}] "
                          f"(no improvement for {early_stopping_patience} epochs)")
                break
        
        training_time = time.time() - start_time
        training_history['training_time'] = training_time
        training_history['best_epoch'] = self.best_epoch
        training_history['best_val_accuracy'] = self.best_val_accuracy
        training_history['best_val_loss'] = self.best_val_loss
        training_history['best_val_f1_macro'] = self.best_val_f1_macro
        training_history['best_val_f1_weighted'] = self.best_val_f1_weighted
        training_history['best_val_balanced_accuracy'] = self.best_val_balanced_accuracy
        training_history['best_epoch_f1_macro'] = self.best_epoch_f1_macro
        training_history['best_epoch_f1_weighted'] = self.best_epoch_f1_weighted
        training_history['best_epoch_balanced_accuracy'] = self.best_epoch_balanced_accuracy
        training_history['epochs_since_improvement'] = self.epochs_since_improvement
        
        # Save training history
        lang_dir = os.path.join(self.experiment_dir, self.language)
        history_path = os.path.join(lang_dir, 'training_history.json')
        write_json(Path(history_path), training_history)
        
        # Save best metrics to a standard JSON file for easy extraction
        metrics_best = {
            'best_val_loss': float(self.best_val_loss),
            'best_val_accuracy': float(self.best_val_accuracy),
            'best_val_f1_macro': float(self.best_val_f1_macro),
            'best_val_f1_weighted': float(self.best_val_f1_weighted),
            'best_val_balanced_accuracy': float(self.best_val_balanced_accuracy),
            'best_epoch': int(self.best_epoch),
            'best_epoch_f1_macro': int(self.best_epoch_f1_macro),
            'best_epoch_f1_weighted': int(self.best_epoch_f1_weighted),
            'best_epoch_balanced_accuracy': int(self.best_epoch_balanced_accuracy),
            'language': self.language,
            'training_time': float(training_time)
        }
        metrics_best_path = os.path.join(lang_dir, 'metrics_best.json')
        write_json(Path(metrics_best_path), metrics_best)
        logger.info(f"Saved best metrics to {metrics_best_path}")
        
        logger.info(f"Training completed [{self.language}] in {training_time:.2f} seconds")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch + 1}")
        logger.info(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        logger.info(f"Best validation F1-macro: {self.best_val_f1_macro:.4f}")
        
        return training_history


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def build_model_config_from_models_section(
    models_cfg: Dict[str, Any],
    language: str,
    base_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a model_config dict compatible with build_model_and_tokenizer()
    from the models: section in the base config.
    
    Args:
        models_cfg: The models: section from base config (e.g., {'en': {'name': ..., 'head_type': ...}})
        language: Language code ('en', 'hi', or 'pa')
        base_config: Full base configuration
    
    Returns:
        Model config dict in format expected by build_model_and_tokenizer()
    """
    if language not in models_cfg:
        raise ValueError(f"Language '{language}' not found in models: section of config")
    
    lang_model_cfg = models_cfg[language]
    model_name = lang_model_cfg['name']
    head_type = lang_model_cfg.get('head_type', 'linear')
    
    # Get max_length from base config or default
    max_length = base_config.get('data', {}).get('max_seq_len', 256)
    
    # Build model config structure expected by build_model_and_tokenizer
    model_config = {
        'model': {
            'name': model_name,
            'tokenizer': {
                'name': model_name,  # Use same name for tokenizer
                'max_length': max_length
            },
            'head_type': head_type
        }
    }
    
    # Add HPA-specific config if head_type is 'hpa'
    if head_type == 'hpa':
        # Read HPA config from models.{lang}.hpa section
        hpa_config = lang_model_cfg.get('hpa', {})
        model_config['model']['head'] = {
            'hidden_size': hpa_config.get('hidden_size', 768),
            'adapter_hidden_sizes': hpa_config.get('hidden_sizes', [512]),  # List of hidden sizes
            'adapter_dropout': hpa_config.get('dropout', 0.2)
        }
        model_config['model']['backbone'] = {
            'checkpoint': model_name
        }
    else:
        # For linear heads, add dropout config
        model_config['model']['head'] = {
            'dropout': lang_model_cfg.get('dropout', 0.1)
        }
    
    return model_config


def build_per_language_label_mapping(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    language: str
) -> Dict[str, Any]:
    """
    Build label mapping from language-specific train split only.
    
    Args:
        train_df: Training dataframe (already filtered by language)
        val_df: Validation dataframe (already filtered by language)
        language: Language code
    
    Returns:
        Label info dict with label2id, id2label, num_labels
    """
    # Get unique labels from train split only (not global)
    train_labels = sorted(train_df['label'].unique())
    val_labels = sorted(val_df['label'].unique())
    
    # Warn if val has labels not in train
    val_only_labels = set(val_labels) - set(train_labels)
    if val_only_labels:
        logger.warning(f"[{language}] Validation has labels not in training: {val_only_labels}")
        logger.warning(f"[{language}] These labels will be mapped to -1 (will be filtered)")
    
    # Build mapping from train labels only
    label2id = {label: idx for idx, label in enumerate(train_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    label_info = {
        'labels': train_labels,
        'label2id': label2id,
        'id2label': id2label,
        'num_labels': len(train_labels),
        'language': language
    }
    
    logger.info(f"[{language}] Built per-language label mapping:")
    logger.info(f"  Unique labels in train: {len(train_labels)}")
    logger.info(f"  Unique labels in val: {len(val_labels)}")
    logger.info(f"  Train labels: {train_labels}")
    logger.info(f"  num_labels: {len(train_labels)}")
    
    return label_info


def train_language(
    language: str,
    model_config: Dict[str, Any],
    base_config: Dict[str, Any],
    label_info: Dict[str, Any],  # Global label info (for reference, but we'll rebuild per-language)
    device: torch.device,
    experiment_dir: str
) -> Dict[str, Any]:
    """
    Train model for a specific language.
    
    Args:
        language: Language code ('en', 'hi', or 'pa')
        model_config: Model-specific configuration (compatible with build_model_and_tokenizer)
        base_config: Base configuration
        label_info: Global label mapping information (for reference)
        device: Training device
        experiment_dir: Experiment directory
    
    Returns:
        Training history dictionary
    """
    model_name = model_config['model']['name']
    head_type = model_config['model'].get('head_type', 'linear')
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training for language: {language}")
    logger.info(f"Loaded model {model_name} for language {language} with head {head_type}")
    logger.info(f"{'='*60}\n")
    
    # Load data with language filtering FIRST
    processed_dir = base_config['paths']['data_processed']
    train_path = os.path.join(processed_dir, 'train.csv')
    val_path = os.path.join(processed_dir, 'val.csv')
    test_path = os.path.join(processed_dir, 'test.csv')
    
    # Load and filter dataframes by language
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    train_df_lang = train_df[train_df['language'] == language].copy()
    val_df_lang = val_df[val_df['language'] == language].copy()
    test_df_lang = test_df[test_df['language'] == language].copy()
    
    # Build per-language label mapping from train split only
    lang_label_info = build_per_language_label_mapping(train_df_lang, val_df_lang, language)
    
    # Save per-language label mapping to experiment folder
    lang_dir = os.path.join(experiment_dir, language)
    os.makedirs(lang_dir, exist_ok=True)
    label_map_path = os.path.join(lang_dir, 'label_map.json')
    write_json(Path(label_map_path), lang_label_info)
    logger.info(f"Saved per-language label mapping to {label_map_path}")
    
    # Debug prints: unique labels and majority class
    unique_labels_train = sorted(train_df_lang['label'].unique())
    unique_labels_val = sorted(val_df_lang['label'].unique())
    
    # Calculate majority class ratio
    train_label_counts = train_df_lang['label'].value_counts()
    majority_class = train_label_counts.index[0]
    majority_count = train_label_counts.iloc[0]
    majority_ratio = majority_count / len(train_df_lang)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"DEBUG: Label Distribution for {language}")
    logger.info(f"{'='*60}")
    logger.info(f"Unique labels in train: {len(unique_labels_train)}")
    logger.info(f"Unique labels in val: {len(unique_labels_val)}")
    logger.info(f"num_labels (from mapping): {lang_label_info['num_labels']}")
    logger.info(f"Train label counts:\n{train_label_counts}")
    logger.info(f"Majority class: '{majority_class}' ({majority_count}/{len(train_df_lang)} = {majority_ratio:.2%})")
    logger.info(f"{'='*60}\n")
    
    # Warn if majority class is >80%
    if majority_ratio > 0.8:
        logger.warning(f"⚠️  MAJORITY CLASS COLLAPSE DETECTED!")
        logger.warning(f"   Majority class '{majority_class}' represents {majority_ratio:.2%} of training data")
        logger.warning(f"   Consider using weighted sampling or focal loss")
    
    # Merge base config with model config for training settings
    config = base_config.copy()
    config['model'] = model_config['model']
    
    # Get max sequence length
    max_seq_len = config.get('data', {}).get('max_seq_len', 256)
    if 'model' in config and 'tokenizer' in config['model']:
        max_seq_len = config['model']['tokenizer'].get('max_length', max_seq_len)
    
    # Build model and tokenizer with per-language label info
    model, tokenizer = build_model_and_tokenizer(
        model_config=config,
        label_info=lang_label_info,  # Use per-language mapping
        device=device
    )
    
    # Verify model is actually loaded correctly (STRICT VALIDATION)
    logger.info(f"\n{'='*60}")
    logger.info(f"MODEL VERIFICATION for {language}")
    logger.info(f"{'='*60}")
    
    # Get expected model name from config
    expected_model_name = model_config['model']['name']
    expected_head_type = model_config['model'].get('head_type', 'linear')
    
    # Extract actual model information
    actual_model_name = None
    actual_head_type = None
    
    if hasattr(model, 'encoder'):
        encoder_type = type(model.encoder).__name__
        if hasattr(model.encoder, 'config'):
            actual_model_name = getattr(model.encoder.config, '_name_or_path', None)
            logger.info(f"Encoder type: {encoder_type}")
            logger.info(f"Model path (from config): {actual_model_name}")
        else:
            logger.info(f"Encoder type: {encoder_type}")
            logger.info(f"Model path: N/A (no config)")
    elif hasattr(model, 'backbone'):
        backbone_type = type(model.backbone).__name__
        if hasattr(model.backbone, 'config'):
            actual_model_name = getattr(model.backbone.config, '_name_or_path', None)
        logger.info(f"Backbone type: {backbone_type}")
        logger.info(f"Model path (from config): {actual_model_name}")
    else:
        model_type = type(model).__name__
        logger.info(f"Model type: {model_type}")
    
    # Determine head type from model structure
    if hasattr(model, 'adapter_head'):
        actual_head_type = 'hpa'
    elif hasattr(model, 'classifier_head'):
        actual_head_type = 'adapter' if hasattr(model.classifier_head, 'adapter_dense') else 'linear'
    else:
        actual_head_type = 'linear'
    
    logger.info(f"Expected backbone: {expected_model_name}")
    logger.info(f"Expected head type: {expected_head_type}")
    logger.info(f"Actual head type: {actual_head_type}")
    
    # HARD ASSERT: Fail fast if backbone doesn't match
    if actual_model_name is None:
        error_msg = (
            f"❌ CRITICAL: Cannot determine model backbone for {language}!\n"
            f"   Expected: {expected_model_name}\n"
            f"   Model structure does not expose _name_or_path.\n"
            f"   This indicates a model loading bug."
        )
        logger.error(error_msg)
        raise AssertionError(error_msg)
    
    if actual_model_name != expected_model_name:
        error_msg = (
            f"❌ HARD ASSERT FAILED: BACKBONE MISMATCH for {language}!\n"
            f"   Expected: {expected_model_name}\n"
            f"   Actual: {actual_model_name}\n"
            f"   This indicates a config parsing error or model loading bug.\n"
            f"   Training will NOT proceed with incorrect model."
        )
        logger.error(error_msg)
        raise AssertionError(error_msg)
    
    # HARD ASSERT: Fail fast if head type doesn't match
    if actual_head_type != expected_head_type:
        error_msg = (
            f"❌ HARD ASSERT FAILED: HEAD TYPE MISMATCH for {language}!\n"
            f"   Expected: {expected_head_type}\n"
            f"   Actual: {actual_head_type}\n"
            f"   This indicates a config parsing error.\n"
            f"   Training will NOT proceed with incorrect head type."
        )
        logger.error(error_msg)
        raise AssertionError(error_msg)
    
    logger.info(f"✓ Backbone verification: PASSED (HARD ASSERT)")
    logger.info(f"✓ Head type verification: PASSED (HARD ASSERT)")
    
    # HARD ASSERT: Fail fast if num_labels doesn't match
    logger.info(f"Model num_labels: {model.num_labels if hasattr(model, 'num_labels') else 'N/A'}")
    logger.info(f"Expected num_labels: {lang_label_info['num_labels']}")
    if not hasattr(model, 'num_labels'):
        error_msg = (
            f"❌ HARD ASSERT FAILED: Model does not have num_labels attribute for {language}!\n"
            f"   Expected: {lang_label_info['num_labels']}\n"
            f"   This indicates a model structure issue."
        )
        logger.error(error_msg)
        raise AssertionError(error_msg)
    
    if model.num_labels != lang_label_info['num_labels']:
        error_msg = (
            f"❌ HARD ASSERT FAILED: NUM_LABELS MISMATCH for {language}!\n"
            f"   Model has {model.num_labels} labels but dataset has {lang_label_info['num_labels']}\n"
            f"   This indicates a label mapping error.\n"
            f"   Training will NOT proceed with incorrect label count."
        )
        logger.error(error_msg)
        raise AssertionError(error_msg)
    
    logger.info(f"✓ Num labels verification: PASSED (HARD ASSERT)")
    logger.info(f"{'='*60}\n")
    
    # Create datasets with language filtering and per-language label mapping
    train_dataset = MultilingualClinicalDataset(
        train_path, tokenizer, lang_label_info['label2id'], 
        max_length=max_seq_len, language=language
    )
    val_dataset = MultilingualClinicalDataset(
        val_path, tokenizer, lang_label_info['label2id'],
        max_length=max_seq_len, language=language
    )
    
    # Create data collator with dynamic padding
    data_collator = DataCollator(tokenizer, padding='longest')
    
    # Get batch size (prefer batch_size, fallback to batch_size_effective for backward compatibility)
    batch_size = int(config['training'].get('batch_size', 
                                            config['training'].get('batch_size_effective', 32)))
    gradient_accumulation_steps = int(config['training'].get('gradient_accumulation_steps', 1))
    logger.info(f"Batch size: {batch_size}, Gradient accumulation steps: {gradient_accumulation_steps}")
    
    # Check if weighted sampling is enabled
    class_sampling = config['training'].get('class_sampling', 'uniform').lower()
    use_weighted_sampling = class_sampling == 'weighted'
    
    # Create data loaders
    if use_weighted_sampling:
        # Calculate sample weights based on inverse frequency (per language)
        train_labels = [lang_label_info['label2id'].get(row['label'], -1)
                       for _, row in train_dataset.data.iterrows()]
        # Filter out -1 (labels not in mapping)
        train_labels = [l for l in train_labels if l != -1]
        
        # Calculate weights per label
        from collections import Counter
        label_counts = Counter(train_labels)
        total_samples = len(train_labels)
        
        # Create sample weights: inverse frequency (normalized)
        # Weight = total_samples / (num_classes * class_count)
        num_classes = len(set(train_labels))
        sample_weights = []
        for label in train_labels:
            count = label_counts[label]
            weight = total_samples / (num_classes * count)  # Inverse frequency
            sample_weights.append(weight)
        
        sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        logger.info(f"Using weighted random sampling (inverse frequency) for {language}")
        logger.info(f"Sample weight range: {sample_weights.min():.4f} - {sample_weights.max():.4f}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,  # Use sampler instead of shuffle
            num_workers=4,
            collate_fn=data_collator
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=data_collator
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=data_collator
    )
    
    logger.info(f"Train samples [{language}]: {len(train_dataset)}")
    logger.info(f"Val samples [{language}]: {len(val_dataset)}")
    
    # Calculate class weights if enabled (use per-language label info)
    # Also enable weighted sampling or focal loss if majority class collapse detected
    class_weights = None
    loss_config = config['training'].get('loss', {})
    
    # Auto-enable weighted sampling if majority class >80%
    if majority_ratio > 0.8 and not use_weighted_sampling:
        logger.warning(f"Auto-enabling weighted sampling due to majority class collapse")
        use_weighted_sampling = True
        class_sampling = 'weighted'
    
    # Auto-switch early stopping to macro F1 if majority class collapse
    early_stopping_config = config['training'].get('early_stopping', {})
    if majority_ratio > 0.8 and early_stopping_config.get('monitor', 'val_loss') == 'val_loss':
        logger.warning(f"Auto-switching early stopping monitor to val_f1_macro due to majority class collapse")
        if 'early_stopping' not in config['training']:
            config['training']['early_stopping'] = {}
        config['training']['early_stopping']['monitor'] = 'val_f1_macro'
    
    if loss_config.get('class_weights') == 'inverse_frequency':
        train_labels = [lang_label_info['label2id'].get(row['label'], -1)
                       for _, row in train_dataset.data.iterrows()]
        # Filter out -1 (labels not in mapping)
        train_labels = [l for l in train_labels if l != -1]
        num_classes = len(lang_label_info['labels'])
        smoothing = float(loss_config.get('class_weights_smoothing', 0.1))
        class_weights = calculate_class_weights_inverse_frequency(
            train_labels, num_classes, smoothing
        ).to(device)
        logger.info(f"Calculated class weights for {language}: {class_weights}")
    
    # Setup optimizer with discriminative LRs
    # Read optimizer type from config (default: "adamw")
    optimizer_name = config['training'].get('optimizer', 'adamw').lower()
    base_lr = float(config['training'].get('learning_rate', 3e-5))
    weight_decay = float(config['training'].get('weight_decay', 0.01))
    no_decay = config['training'].get('no_decay_on', ['bias', 'LayerNorm'])
    
    param_groups = get_optimizer_param_groups(model, base_lr, weight_decay, no_decay)
    
    # Create optimizer based on config
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(param_groups, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        sgd_momentum = float(config['training'].get('sgd_momentum', 0.9))
        optimizer = torch.optim.SGD(param_groups, momentum=sgd_momentum, weight_decay=weight_decay)
    else:
        logger.warning(f"Unknown optimizer '{optimizer_name}', defaulting to AdamW")
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        optimizer_name = 'adamw'
    
    # Setup scheduler with warmup
    # Read epochs (prefer 'epochs', fallback to 'max_epochs' for backward compatibility)
    epochs = int(config['training'].get('epochs', config['training'].get('max_epochs', 10)))
    num_training_steps = len(train_loader) * epochs
    
    # Read warmup_steps (if set, use it; otherwise calculate from warmup_ratio)
    warmup_steps_config = config['training'].get('warmup_steps')
    if warmup_steps_config is not None:
        warmup_steps = int(warmup_steps_config)
    else:
        warmup_ratio = float(config['training'].get('warmup_ratio', 0.05))
        warmup_steps = int(num_training_steps * warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    logger.info(f"Optimizer: {optimizer_name.upper()}, LR: {base_lr}, Weight decay: {weight_decay}")
    if optimizer_name == 'sgd':
        logger.info(f"SGD momentum: {config['training'].get('sgd_momentum', 0.9)}")
    logger.info(f"Scheduler: Linear warmup ({warmup_steps} steps) + decay")
    
    # Label smoothing (enabled by default for better generalization)
    # Can be disabled by setting label_smoothing: false in config
    label_smoothing = 0.05 if loss_config.get('label_smoothing', True) else 0.0
    if label_smoothing > 0:
        logger.info(f"Label smoothing enabled: {label_smoothing}")
    
    # Create trainer (use per-language label info)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        experiment_dir=experiment_dir,
        label_info=lang_label_info,  # Use per-language mapping
        config=config,
        language=language,
        class_weights=class_weights,
        label_smoothing=label_smoothing
    )
    
    # Train
    # Read epochs (prefer 'epochs', fallback to 'max_epochs' for backward compatibility)
    epochs = int(config['training'].get('epochs', config['training'].get('max_epochs', 10)))
    # Read early stopping patience (prefer 'early_stopping_patience', fallback to nested structure)
    early_stopping_config = config['training'].get('early_stopping', {})
    early_stopping_patience = int(config['training'].get('early_stopping_patience', 
                                                          early_stopping_config.get('patience', 3)))
    
    training_history = trainer.train(epochs, early_stopping_patience)
    
    return training_history


def compute_overall_summary(
    all_results: Dict[str, Dict[str, Any]],
    experiment_id: str,
    experiment_root: str
) -> Dict[str, Any]:
    """
    Compute overall summary with per-language metrics and macro-averages.
    
    Args:
        all_results: Dictionary mapping language codes to training history dicts
        experiment_id: Experiment ID
        experiment_root: Root directory for experiments
    
    Returns:
        Dictionary with per-language metrics and overall averages
    """
    per_language = {}
    metric_keys = [
        'best_val_loss', 'best_val_accuracy', 'best_val_f1_macro',
        'best_val_f1_weighted', 'best_val_balanced_accuracy',
        'best_epoch', 'best_epoch_f1_macro', 'best_epoch_f1_weighted',
        'best_epoch_balanced_accuracy', 'training_time'
    ]
    
    for lang, results in all_results.items():
        lang_metrics = {}
        for key in metric_keys:
            lang_metrics[key] = results.get(key, 0.0)
        per_language[lang] = lang_metrics
    
    # Compute macro-averages (simple mean across languages)
    overall_averages = {}
    for key in metric_keys:
        if key in ['best_epoch', 'best_epoch_f1_macro', 'best_epoch_f1_weighted', 
                   'best_epoch_balanced_accuracy']:
            # For epoch metrics, compute mean (can be fractional)
            values = [per_language[lang].get(key, 0) for lang in per_language.keys()]
            overall_averages[key] = float(np.mean(values)) if values else 0.0
        elif key == 'training_time':
            # For time, compute total
            values = [per_language[lang].get(key, 0) for lang in per_language.keys()]
            overall_averages[key] = float(np.sum(values)) if values else 0.0
        else:
            # For other metrics, compute mean
            values = [per_language[lang].get(key, 0.0) for lang in per_language.keys()]
            overall_averages[key] = float(np.mean(values)) if values else 0.0
    
    summary = {
        'experiment_id': experiment_id,
        'experiment_root': experiment_root,
        'per_language': per_language,
        'overall_averages': overall_averages,
        'num_languages': len(per_language)
    }
    
    return summary


def get_experiment_dir(
    base_config: Dict[str, Any],
    experiment_id: str,
    language: str,
    clean_experiment_dir: bool = False
) -> str:
    """
    Get (and optionally clean) the experiment directory for a language.

    Layout:
      <experiment_root>/
        <experiment_id>/
          <language>/
            best.ckpt, checkpoints/, training_history.json, metrics.json, ...

    Note: If training.checkpoint_dir is set, checkpoints will be saved there
    instead, but experiment_dir still points to experiment_root for other files.

    Args:
        base_config: Loaded base configuration.
        experiment_id: Run identifier (e.g. 'E17636...').
        language: Language code ('en', 'hi', or 'pa').
        clean_experiment_dir: If True, delete existing experiment directory before creating new one.
    """
    # Get experiment_root from training config
    experiment_root = base_config.get('training', {}).get('experiment_root', 'experiments')
    
    # If experiment_root is relative, ensure it's a valid path
    if not os.path.isabs(experiment_root):
        # If it doesn't start with 'experiments', prepend the base experiments path
        if not experiment_root.startswith('experiments'):
            base_experiments = base_config['paths']['experiments']
            experiment_root = os.path.join(base_experiments, experiment_root)
        # Otherwise, use it as-is (it's already like "experiments/xlmroberta")
    
    experiment_dir = os.path.join(experiment_root, experiment_id)
    
    if clean_experiment_dir and os.path.exists(experiment_dir):
        logger.info(f"Cleaning existing experiment directory: {experiment_dir}")
        shutil.rmtree(experiment_dir)
    
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Log checkpoint directory if overridden
    checkpoint_dir = base_config.get('training', {}).get('checkpoint_dir')
    if checkpoint_dir:
        logger.info(f"Checkpoint directory override: {checkpoint_dir}")
        logger.info(f"Checkpoints will be saved to: {os.path.join(checkpoint_dir, experiment_id, language)}")
        logger.info(f"Other files (metrics, history) will be saved to: {experiment_dir}")
    
    return experiment_dir


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train multilingual clinical NLP models')
    parser.add_argument('--config', type=str, default='src/config/base_distilbert.yaml',
                       help='Path to base configuration file (e.g., base_distilbert.yaml, base_indicbert_hpa.yaml)')
    parser.add_argument('--experiment-id', type=str, default=None,
                       help='Experiment ID (default: auto-generated)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, default: auto)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load base config
    base_config = load_config(args.config)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    
    # Load label info
    processed_dir = base_config['paths']['data_processed']
    labels_path = os.path.join(processed_dir, 'labels.json')
    label_info = load_label_mapping(labels_path)
    
    # Setup experiment id
    if args.experiment_id:
        experiment_id = args.experiment_id
    else:
        import time
        experiment_id = f"E{int(time.time())}"

    logger.info(f"Experiment ID: {experiment_id}")

    # Check if config has the new models: section
    if 'models' not in base_config:
        raise ValueError(
            "Config must contain a 'models:' section mapping languages to model configs. "
            "Please use one of the per-backbone config files (base_distilbert.yaml, etc.)"
        )
    
    models_cfg = base_config['models']
    languages = base_config.get('data', {}).get('languages', ['en', 'hi', 'pa'])
    
    # ============================================================
    # MULTILINGUAL TRAINING STRATEGY
    # ============================================================
    # Option A (CURRENT IMPLEMENTATION): Per-language training
    #   - Each language is trained separately with its own model instance
    #   - Label mapping (label2id/id2label) is built per-language from that language's train split
    #   - This allows different languages to have different label sets (e.g., hi has 5 labels, en has 6)
    #   - Each language gets its own experiment directory: <experiment_root>/<exp_id>/<lang>/
    #   - This is the default and recommended approach for this project
    #
    # Option B (NOT IMPLEMENTED): Joint multilingual training
    #   - Would require a canonical label schema across all languages
    #   - All languages would share the same label2id/id2label mapping
    #   - Would train a single model on mixed-language data
    #   - This is NOT currently supported
    # ============================================================
    
    all_results = {}
    
    # Get experiment root directory (will be shared for all languages in this run)
    experiment_root = base_config.get('training', {}).get('experiment_root', 'experiments')
    # If experiment_root is relative, ensure it's a valid path
    if not os.path.isabs(experiment_root):
        # If it doesn't start with 'experiments', prepend the base experiments path
        if not experiment_root.startswith('experiments'):
            base_experiments = base_config['paths']['experiments']
            experiment_root = os.path.join(base_experiments, experiment_root)
        # Otherwise, use it as-is (it's already like "experiments/xlmroberta")
    
    # Train each language with its configured model
    for lang in languages:
        if lang not in models_cfg:
            logger.warning(f"Language '{lang}' is in data.languages but not in models: section. Skipping.")
            continue
        
        # Build model config for this language
        model_config = build_model_config_from_models_section(
            models_cfg=models_cfg,
            language=lang,
            base_config=base_config
        )
        
        # Get experiment directory for this run (shared across languages)
        experiment_dir = get_experiment_dir(
            base_config=base_config,
            experiment_id=experiment_id,
            language=lang,
            clean_experiment_dir=(lang == languages[0])  # Clean only for first language
        )
        
        # Train this language
        lang_results = train_language(
            language=lang,
            model_config=model_config,
            base_config=base_config,
            label_info=label_info,
            device=device,
            experiment_dir=experiment_dir
        )
        all_results[lang] = lang_results
    
    # Compute and save aggregation/averages
    summary_overall = compute_overall_summary(all_results, experiment_id, experiment_root)
    
    # Save overall summary (JSON and CSV)
    summary_path = os.path.join(experiment_root, experiment_id, 'summary_overall.json')
    write_json(Path(summary_path), summary_overall)
    logger.info(f"Saved overall summary to {summary_path}")
    
    # Save CSV summary
    summary_csv_path = os.path.join(experiment_root, experiment_id, 'summary_overall.csv')
    summary_df = pd.DataFrame(summary_overall['per_language']).T
    summary_df.to_csv(summary_csv_path, index=True)
    logger.info(f"Saved overall summary CSV to {summary_csv_path}")

    logger.info("\n" + "="*60)
    logger.info("Training completed for all languages!")
    logger.info("="*60)
    for lang, results in all_results.items():
        logger.info(f"{lang.upper()}: Best val loss = {results['best_val_loss']:.4f}, "
                   f"Best val accuracy = {results['best_val_accuracy']:.4f}, "
                   f"Best F1-macro = {results.get('best_val_f1_macro', 0.0):.4f}")
    
    logger.info("\n" + "="*60)
    logger.info("OVERALL AVERAGES (Macro across languages):")
    logger.info("="*60)
    overall_avg = summary_overall['overall_averages']
    logger.info(f"  Accuracy: {overall_avg.get('accuracy', 0.0):.4f}")
    logger.info(f"  Balanced Accuracy: {overall_avg.get('balanced_accuracy', 0.0):.4f}")
    logger.info(f"  F1-macro: {overall_avg.get('f1_macro', 0.0):.4f}")
    logger.info(f"  F1-weighted: {overall_avg.get('f1_weighted', 0.0):.4f}")
    logger.info(f"  Loss: {overall_avg.get('loss', 0.0):.4f}")


if __name__ == '__main__':
    main()
