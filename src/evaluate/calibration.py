"""
Calibration evaluation for multi-class classification.

This module implements Expected Calibration Error (ECE) to assess the reliability
of diagnostic probabilities. Calibration is crucial for clinical applications
where probability estimates need to reflect true likelihoods.

Expected Calibration Error (ECE) measures the difference between predicted
confidence and actual accuracy. A well-calibrated model should have confidence
scores that match the true probability of correctness.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def expected_calibration_error(
    y_true: Union[np.ndarray, List[int]],
    y_proba: Union[np.ndarray, List[List[float]]],
    n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error (ECE) for multi-class classification.
    
    ECE measures the difference between predicted confidence and actual accuracy.
    For each sample:
    - Predicted class = argmax(probabilities)
    - Confidence = max(probabilities)
    
    Predictions are binned by confidence into n_bins equally spaced bins (0-1).
    For each bin:
    - Average confidence = mean of confidences in bin
    - Accuracy = fraction of correct predictions in bin
    
    ECE = Σ (bin_size / N) * |accuracy - confidence| over all bins
    
    A lower ECE indicates better calibration. ECE = 0 means perfect calibration.
    
    Args:
        y_true: Ground truth labels (array-like of shape [n_samples])
        y_proba: Predicted probabilities (array-like of shape [n_samples, n_classes])
        n_bins: Number of bins for calibration (default: 15)
    
    Returns:
        Expected Calibration Error (float, 0-1 range)
    
    References:
        Guo, C., et al. "On Calibration of Modern Neural Networks." ICML 2017.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    # Validate inputs
    if y_proba.ndim != 2:
        raise ValueError(f"y_proba must be 2D array, got shape {y_proba.shape}")
    
    if len(y_true) != y_proba.shape[0]:
        raise ValueError(
            f"y_true and y_proba must have same length: "
            f"{len(y_true)} vs {y_proba.shape[0]}"
        )
    
    n_samples = len(y_true)
    
    # Get predicted classes and confidences
    predicted_classes = np.argmax(y_proba, axis=1)
    confidences = np.max(y_proba, axis=1)
    
    # Check if predictions are correct
    correct = (predicted_classes == y_true).astype(float)
    
    # Bin edges (equally spaced from 0 to 1)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    # Assign each prediction to a bin
    # Use right=False so bins are [lower, upper) except last bin which is [lower, upper]
    # For confidence = 1.0, we want it in the last bin
    bin_indices = np.digitize(confidences, bin_edges, right=False) - 1
    
    # Handle edge case: confidence = 1.0 should go to last bin
    # np.digitize with right=False puts 1.0 in bin n_bins+1, we clip to n_bins-1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Explicitly handle confidence = 1.0 to ensure it's in the last bin
    bin_indices[confidences == 1.0] = n_bins - 1
    
    # Compute ECE
    ece = 0.0
    
    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (bin_indices == i)
        bin_size = np.sum(in_bin)
        
        if bin_size == 0:
            # Skip empty bins
            continue
        
        # Average confidence in this bin
        bin_confidence = np.mean(confidences[in_bin])
        
        # Accuracy in this bin
        bin_accuracy = np.mean(correct[in_bin])
        
        # Weighted contribution to ECE
        ece += (bin_size / n_samples) * abs(bin_accuracy - bin_confidence)
    
    return float(ece)


def calibration_summary(
    y_true: Union[np.ndarray, List[int]],
    y_proba: Union[np.ndarray, List[List[float]]],
    n_bins: int = 15
) -> Dict[str, Any]:
    """
    Compute detailed calibration summary including per-bin statistics.
    
    This function provides a comprehensive view of model calibration by
    analyzing confidence-accuracy alignment across different confidence levels.
    This is essential for clinical applications where probability estimates
    must be reliable for diagnostic decision-making.
    
    Args:
        y_true: Ground truth labels (array-like of shape [n_samples])
        y_proba: Predicted probabilities (array-like of shape [n_samples, n_classes])
        n_bins: Number of bins for calibration (default: 15)
    
    Returns:
        Dictionary containing:
        - 'ece': Overall Expected Calibration Error
        - 'bins': List of dictionaries, one per bin, containing:
          - 'interval': Tuple (lower, upper) confidence interval
          - 'accuracy': Accuracy in this bin
          - 'avg_confidence': Average confidence in this bin
          - 'count': Number of samples in this bin
        - 'n_samples': Total number of samples
        - 'n_bins': Number of bins used
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    # Validate inputs
    if y_proba.ndim != 2:
        raise ValueError(f"y_proba must be 2D array, got shape {y_proba.shape}")
    
    if len(y_true) != y_proba.shape[0]:
        raise ValueError(
            f"y_true and y_proba must have same length: "
            f"{len(y_true)} vs {y_proba.shape[0]}"
        )
    
    n_samples = len(y_true)
    
    # Get predicted classes and confidences
    predicted_classes = np.argmax(y_proba, axis=1)
    confidences = np.max(y_proba, axis=1)
    
    # Check if predictions are correct
    correct = (predicted_classes == y_true).astype(float)
    
    # Bin edges (equally spaced from 0 to 1)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    # Assign each prediction to a bin
    # Use right=False so bins are [lower, upper) except last bin which is [lower, upper]
    bin_indices = np.digitize(confidences, bin_edges, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Explicitly handle confidence = 1.0 to ensure it's in the last bin
    bin_indices[confidences == 1.0] = n_bins - 1
    
    # Compute per-bin statistics
    bins = []
    
    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (bin_indices == i)
        bin_size = int(np.sum(in_bin))
        
        # Bin interval
        lower = float(bin_edges[i])
        upper = float(bin_edges[i + 1])
        
        if bin_size == 0:
            # Empty bin
            bins.append({
                'interval': (lower, upper),
                'accuracy': None,
                'avg_confidence': None,
                'count': 0
            })
        else:
            # Average confidence in this bin
            bin_confidence = float(np.mean(confidences[in_bin]))
            
            # Accuracy in this bin
            bin_accuracy = float(np.mean(correct[in_bin]))
            
            bins.append({
                'interval': (lower, upper),
                'accuracy': bin_accuracy,
                'avg_confidence': bin_confidence,
                'count': bin_size
            })
    
    # Compute overall ECE
    ece = expected_calibration_error(y_true, y_proba, n_bins)
    
    result = {
        'ece': ece,
        'bins': bins,
        'n_samples': n_samples,
        'n_bins': n_bins
    }
    
    logger.info(f"Expected Calibration Error: {ece:.4f}")
    logger.info(f"Calibration bins: {n_bins}, Samples: {n_samples}")
    
    return result


def plot_calibration_curve(
    y_true: Union[np.ndarray, List[int]],
    y_proba: Union[np.ndarray, List[List[float]]],
    n_bins: int = 15,
    save_path: Optional[str] = None
) -> None:
    """
    Plot calibration curve (reliability diagram).
    
    Creates a visualization showing the relationship between predicted
    confidence and actual accuracy. A perfectly calibrated model would
    have points along the diagonal (confidence = accuracy).
    
    Args:
        y_true: Ground truth labels
        y_proba: Predicted probabilities
        n_bins: Number of bins for calibration
        save_path: Optional path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping calibration plot")
        return
    
    summary = calibration_summary(y_true, y_proba, n_bins)
    
    # Extract bin statistics
    bin_centers = []
    accuracies = []
    confidences = []
    counts = []
    
    for bin_info in summary['bins']:
        if bin_info['count'] > 0:
            lower, upper = bin_info['interval']
            bin_center = (lower + upper) / 2
            bin_centers.append(bin_center)
            accuracies.append(bin_info['accuracy'])
            confidences.append(bin_info['avg_confidence'])
            counts.append(bin_info['count'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot calibration curve
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    ax.plot(confidences, accuracies, 'o-', label='Model', linewidth=2, markersize=8)
    
    # Add bin size as bar width (optional visualization)
    # Normalize counts for visualization
    if counts:
        max_count = max(counts)
        widths = [c / max_count * 0.05 for c in counts]  # Scale to reasonable bar width
        ax.bar(confidences, accuracies, width=widths, alpha=0.3, label='Bin size')
    
    ax.set_xlabel('Mean Predicted Confidence', fontsize=12)
    ax.set_ylabel('Fraction of Positives (Accuracy)', fontsize=12)
    ax.set_title(f'Calibration Curve (ECE = {summary["ece"]:.4f})', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Calibration curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

