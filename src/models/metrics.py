import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report,
    roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef
)
from sklearn.exceptions import UndefinedMetricWarning
from typing import Dict, List, Any, Tuple, Optional, Union
import logging
import torch
import warnings

# Suppress warnings about classes not present in y_true
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true')

logger = logging.getLogger(__name__)

class MetricCalculator:
    """Calculate metrics for model evaluation."""
    
    def __init__(self, id2label: Dict[int, str], include_weighted: bool = False):
        # Convert all keys to integers (handles case where labels.json has string keys)
        id2label = {int(k): v for k, v in id2label.items()}
        self.id2label = id2label
        self.labels = [id2label[i] for i in range(len(id2label))]
        self.include_weighted = include_weighted
    
    def compute_metrics(self, eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Compute metrics for evaluation predictions."""
        predictions, labels = eval_pred
        
        # Convert to numpy arrays (handles case where inputs are Python lists)
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)
        
        # Handle logits if provided (2D array with shape [n_samples, n_classes])
        if predictions.ndim > 1 and predictions.shape[1] > 1:
            preds = predictions.argmax(-1)
            # Use raw predictions (logits/probabilities) for AUROC, AUPRC, ECE
            logits_or_probs = predictions
        else:
            preds = predictions
            # For 1D predictions, we can't compute AUROC, AUPRC, ECE without probabilities
            logits_or_probs = None
        
        # Compute metrics using numpy arrays
        accuracy = accuracy_score(labels, preds)
        balanced_acc = balanced_accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
        precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
        recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
        
        # Compute weighted metrics if requested
        if self.include_weighted:
            f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
            precision_weighted = precision_score(labels, preds, average='weighted', zero_division=0)
            recall_weighted = recall_score(labels, preds, average='weighted', zero_division=0)
        
        # MCC (Matthews Correlation Coefficient)
        try:
            mcc = matthews_corrcoef(labels, preds)
        except Exception as e:
            logger.warning(f"Could not compute MCC: {e}")
            mcc = 0.0
        
        metrics = {
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_acc),
            'f1_macro': float(f1_macro),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'mcc': float(mcc)
        }
        
        # Add weighted metrics if requested
        if self.include_weighted:
            metrics.update({
                'f1_weighted': float(f1_weighted),
                'precision_weighted': float(precision_weighted),
                'recall_weighted': float(recall_weighted)
            })
        
        # Compute additional metrics if logits/probabilities are available
        if logits_or_probs is not None:
            # Compute AUROC
            try:
                auroc_scores = compute_auroc_ovr(labels, logits_or_probs)
                metrics.update({
                    'auroc_macro': float(auroc_scores['macro'])
                })
                if self.include_weighted:
                    # For weighted AUROC, we need to compute it separately
                    try:
                        auroc_weighted = roc_auc_score(labels, logits_or_probs, multi_class='ovr', average='weighted')
                        metrics.update({
                            'auroc_weighted': float(auroc_weighted)
                        })
                    except Exception as e:
                        logger.warning(f"Could not compute weighted AUROC: {e}")
            except Exception as e:
                logger.warning(f"Could not compute AUROC: {e}")
            
            # Compute AUPRC
            try:
                auprc_scores = compute_auprc_ovr(labels, logits_or_probs)
                metrics.update({
                    'auprc_macro': float(auprc_scores['macro'])
                })
                if self.include_weighted:
                    # For weighted AUPRC, we need to compute it separately
                    try:
                        auprc_weighted = average_precision_score(labels, logits_or_probs, average='weighted')
                        metrics.update({
                            'auprc_weighted': float(auprc_weighted)
                        })
                    except Exception as e:
                        logger.warning(f"Could not compute weighted AUPRC: {e}")
            except Exception as e:
                logger.warning(f"Could not compute AUPRC: {e}")
            
            # Compute ECE
            try:
                ece = compute_ece(labels, logits_or_probs)
                metrics['ece'] = float(ece)
            except Exception as e:
                logger.warning(f"Could not compute ECE: {e}")
        
        return metrics
    
    def compute_per_language_metrics(
        self, 
        predictions: List[int], 
        labels: List[int], 
        languages: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics per language."""
        unique_languages = set(languages)
        language_metrics = {}
        
        for lang in unique_languages:
            lang_mask = [l == lang for l in languages]
            lang_preds = [p for p, m in zip(predictions, lang_mask) if m]
            lang_labels = [l for l, m in zip(labels, lang_mask) if m]
            
            if len(lang_preds) == 0:
                continue
                
            lang_accuracy = accuracy_score(lang_labels, lang_preds)
            lang_f1 = f1_score(lang_labels, lang_preds, average='macro')
            lang_precision = precision_score(lang_labels, lang_preds, average='macro')
            lang_recall = recall_score(lang_labels, lang_preds, average='macro')
            
            language_metrics[lang] = {
                'accuracy': lang_accuracy,
                'f1_macro': lang_f1,
                'precision_macro': lang_precision,
                'recall_macro': lang_recall,
                'support': len(lang_preds)
            }
            
            logger.info(f"Language {lang} metrics: accuracy={lang_accuracy:.4f}, f1_macro={lang_f1:.4f}")
        
        return language_metrics
    
    def compute_detailed_metrics(
        self,
        predictions: List[int],
        labels: List[int],
        languages: List[str]
    ) -> Dict[str, Any]:
        """Compute comprehensive metrics including per-language and per-class."""
        
        # Convert to numpy arrays
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)
        
        # Overall metrics (reuse compute_metrics)
        overall_metrics = self.compute_metrics((predictions, labels))
        
        # Find label IDs that actually appear in this run
        present_label_ids = np.unique(labels)
        present_label_ids = np.sort(present_label_ids)
        
        # Build matching target_names from self.labels
        target_names = [self.id2label[i] for i in present_label_ids]
        
        # Compute classification report
        class_report = classification_report(
            labels, predictions, 
            labels=present_label_ids,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        
        # Remove weighted average metrics from the classification report if not requested
        if not self.include_weighted and 'weighted avg' in class_report:
            del class_report['weighted avg']
        
        # Per-language metrics
        per_language = self.compute_per_language_metrics(predictions.tolist(), labels.tolist(), languages)
        
        return {
            'overall': overall_metrics,
            'per_class': class_report,
            'per_language': per_language
        }


def compute_metrics_from_logits(
    logits: np.ndarray, 
    labels: np.ndarray,
    label2id: Dict[str, int],
    include_weighted: bool = False
) -> Dict[str, float]:
    """Compute metrics directly from logits and labels."""
    predictions = logits.argmax(-1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro')
    precision_macro = precision_score(labels, predictions, average='macro')
    recall_macro = recall_score(labels, predictions, average='macro')
    
    metrics = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro)
    }
    
    if include_weighted:
        f1_weighted = f1_score(labels, predictions, average='weighted')
        precision_weighted = precision_score(labels, predictions, average='weighted')
        recall_weighted = recall_score(labels, predictions, average='weighted')
        metrics.update({
            'f1_weighted': float(f1_weighted),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted)
        })
    
    return metrics


def compute_auroc_ovr(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int] = None) -> Dict[str, float]:
    """Compute Area Under Receiver Operating Characteristic (AUROC) using one-vs-rest strategy."""
    try:
        # Convert to numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Check if y_pred is logits (2D) or probabilities (2D) or predictions (1D)
        if y_pred.ndim == 1 or y_pred.shape[1] == 1:
            # Single-class or binary classification with single output
            # Convert to one-vs-rest probabilities if needed
            if y_pred.ndim == 1:
                # Convert 1D predictions to probabilities (assuming binary)
                y_pred = np.stack([1 - y_pred, y_pred], axis=1)
            
        # Compute AUROC
        auroc_macro = roc_auc_score(y_true, y_pred, multi_class='ovr', average='macro')
        
        return {
            'macro': float(auroc_macro)
        }
    except Exception as e:
        logger.warning(f"Could not compute AUROC: {e}")
        return {
            'macro': 0.0
        }


def compute_auprc_ovr(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int] = None) -> Dict[str, float]:
    """Compute Area Under Precision-Recall Curve (AUPRC) using one-vs-rest strategy."""
    try:
        # Convert to numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Check if y_pred is logits (2D) or probabilities (2D) or predictions (1D)
        if y_pred.ndim == 1 or y_pred.shape[1] == 1:
            # Single-class or binary classification with single output
            # Convert to one-vs-rest probabilities if needed
            if y_pred.ndim == 1:
                # Convert 1D predictions to probabilities (assuming binary)
                y_pred = np.stack([1 - y_pred, y_pred], axis=1)
        
        # Compute AUPRC
        auprc_macro = average_precision_score(y_true, y_pred, average='macro')
        
        return {
            'macro': float(auprc_macro)
        }
    except Exception as e:
        logger.warning(f"Could not compute AUPRC: {e}")
        return {
            'macro': 0.0
        }


def classification_report_with_macro_f1(
    y_true: List[int], 
    y_pred: List[int], 
    label_names: List[str],
    include_weighted: bool = False
) -> Dict[str, Any]:
    """
    Generate classification report with macro F1 score as the primary metric.
    
    Args:
        y_true: List of true label IDs
        y_pred: List of predicted label IDs
        label_names: List of label names in order
        include_weighted: Whether to include weighted average metrics
    
    Returns:
        Dictionary containing:
        - accuracy: Overall accuracy
        - macro_f1: Macro-averaged F1 score
        - per_class: Per-class metrics
        - macro_avg: Macro-averaged metrics
        - weighted_avg: Weighted-averaged metrics (if include_weighted=True)
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Generate classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )
    
    # Calculate additional metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Prepare results
    results = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'per_class': report,
        'macro_avg': report['macro avg']
    }
    
    # Add weighted average if requested
    if include_weighted and 'weighted avg' in report:
        results['weighted_avg'] = report['weighted avg']
    elif not include_weighted and 'weighted avg' in report:
        # Remove weighted average if not requested
        del report['weighted avg']
    
    return results


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        y_true: True labels (1D array)
        y_prob: Predicted probabilities for the true class (1D array) or class probabilities (2D array)
        n_bins: Number of bins to use for calibration
    
    Returns:
        Expected Calibration Error
    """
    try:
        # Convert to numpy arrays
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        
        # If y_prob is 2D (class probabilities), get probabilities for the true class
        if y_prob.ndim == 2:
            # Get predicted classes
            y_pred = np.argmax(y_prob, axis=1)
            # Get probabilities for predicted classes
            y_prob = np.max(y_prob, axis=1)
        elif y_prob.ndim == 1:
            # y_prob is already probabilities for the positive class (binary case)
            y_pred = (y_prob > 0.5).astype(int)
        else:
            raise ValueError(f"y_prob must be 1D or 2D, got {y_prob.ndim}D")
        
        # Compute calibration bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        total_count = len(y_true)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Determine if probability is in bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                # Compute accuracy in bin
                bin_accuracy = np.mean(y_true[in_bin] == y_pred[in_bin])
                # Compute average confidence in bin
                bin_confidence = np.mean(y_prob[in_bin])
                # Compute ECE contribution
                ece += (bin_size / total_count) * np.abs(bin_accuracy - bin_confidence)
        
        return float(ece)
    except Exception as e:
        logger.warning(f"Could not compute ECE: {e}")
        return 0.0