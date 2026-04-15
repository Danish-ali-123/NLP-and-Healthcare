"""
Zero-shot LLM evaluation for orthopedic diagnostic classification.

This script implements the zero-shot baseline evaluation for GPT-4-family and DeepSeek
large language models, as described in the Research Methodology section of the thesis.

The zero-shot approach evaluates pre-trained LLMs' ability to classify orthopedic
diagnoses without task-specific fine-tuning, providing a baseline comparison against
fine-tuned transformer models (DistilBERT for English, IndicBERT-HPA for Hindi/Punjabi).

This evaluation:
1. Loads the same test set used by fine-tuned models
2. Constructs prompts using a standardized template with label information
3. Calls GPT-4o-mini or DeepSeek via their respective APIs
4. Parses LLM responses to extract predicted label IDs
5. Computes metrics (Macro-F1 as primary, per-class precision/recall/F1)
6. Saves results for comparison with fine-tuned models

Results are saved to:
- experiments/E900_gpt4o_zero_shot/metrics.json
- experiments/E901_deepseek_zero_shot/metrics.json
"""

import argparse
import json
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.llm.gpt_utils import gpt_zero_shot_predict
from src.llm.deepseek_utils import deepseek_zero_shot_predict
from src.data.dataset import load_label_mapping
from src.config import base_config
from src.models.metrics import classification_report_with_macro_f1

logger = logging.getLogger(__name__)


def load_test_data(test_path: str) -> pd.DataFrame:
    """
    Load test data from CSV or JSONL file.
    
    Args:
        test_path: Path to test data file (CSV or JSONL)
    
    Returns:
        DataFrame with columns: text, label, language (and optionally id)
    """
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    # Load based on file extension
    if test_path.endswith('.jsonl'):
        logger.info(f"Loading JSONL data from {test_path}")
        # Load JSONL file
        records = []
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        df = pd.DataFrame(records)
    else:
        # Assume CSV format
        logger.info(f"Loading CSV data from {test_path}")
        df = pd.read_csv(test_path)
    
    # Validate required columns
    required_cols = ['text', 'label', 'language']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Test data missing required columns: {missing_cols}")
    
    # Normalize language codes to lowercase
    df['language'] = df['language'].str.lower()
    # Map full names to codes if needed
    lang_map = {
        'english': 'en',
        'hindi': 'hi',
        'punjabi': 'pa'
    }
    df['language'] = df['language'].map(lang_map).fillna(df['language'])
    
    logger.info(f"Loaded {len(df)} test examples")
    logger.info(f"Language distribution: {df['language'].value_counts().to_dict()}")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df


def load_prompt_template(prompt_path: str) -> str:
    """Load prompt template from file."""
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    return Path(prompt_path).read_text(encoding="utf-8")


def format_label_block(label_info: Dict[str, Any]) -> str:
    """
    Format label information into a multi-line string for the prompt.
    
    Args:
        label_info: Dictionary with 'id2label' mapping
    
    Returns:
        Multi-line string with format "ID - NAME" for each label, sorted by ID
    """
    id2label = label_info['id2label']
    # Convert string keys to int for proper sorting
    sorted_labels = sorted([(int(k), v) for k, v in id2label.items()])
    label_lines = [f"{label_id} - {label_name}" for label_id, label_name in sorted_labels]
    return "\n".join(label_lines)


def get_language_name(lang_code: str) -> str:
    """Convert language code to full name."""
    lang_map = {
        'en': 'English',
        'hi': 'Hindi',
        'pa': 'Punjabi'
    }
    return lang_map.get(lang_code.lower(), lang_code.capitalize())


def render_prompt(template: str, text: str, lang: str, label_info: Dict[str, Any]) -> str:
    """
    Render prompt template with text, language, and label block.
    
    Args:
        template: Prompt template string with placeholders
        text: Clinical note text
        lang: Language code (en, hi, pa)
        label_info: Label information dictionary with id2label mapping
    
    Returns:
        Fully rendered prompt string
    """
    label_block = format_label_block(label_info)
    language_name = get_language_name(lang)
    
    # Replace placeholders
    rendered = template.replace("{label_block}", label_block)
    rendered = rendered.replace("{language}", language_name)
    rendered = rendered.replace("{text}", text)
    
    return rendered


def parse_label_id(response: str) -> Optional[int]:
    """
    Parse label ID from LLM response.
    
    Strips whitespace and punctuation, then extracts the leading integer.
    Handles various response formats like "0", "Label 0", "0.", etc.
    
    Args:
        response: Raw string response from LLM
    
    Returns:
        Integer label ID if successfully parsed, None otherwise
    """
    if not response:
        return None
    
    # Strip whitespace
    response = response.strip()
    
    # Try to extract first integer from the response
    # Match patterns like "0", "0.", "Label 0", "The answer is 0", etc.
    match = re.search(r'\b(\d+)\b', response)
    if match:
        try:
            label_id = int(match.group(1))
            return label_id
        except ValueError:
            pass
    
    return None


def plot_confusion_matrix(
    y_true: List[int], 
    y_pred: List[int], 
    id2label: Dict[int, str], 
    output_path: Path
) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True label IDs
        y_pred: Predicted label IDs  
        id2label: ID to label name mapping
        output_path: Output directory path
    """
    # Import sklearn here to avoid dependency issues if not needed
    from sklearn.metrics import confusion_matrix
    
    # Get sorted unique labels
    labels = sorted(set(y_true + y_pred))
    label_names = [id2label[str(l)] for l in labels]
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save plot
    cm_path = output_path / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    


def plot_classification_report(
    results: Dict[str, Any], 
    output_path: Path
) -> None:
    """
    Plot classification report as heatmap.
    
    Args:
        results: Evaluation results dictionary
        output_path: Output directory path
    """
    # Extract per-class metrics
    per_class = results['per_class_metrics']
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(per_class).T
    
    # Select metrics columns (accuracy, precision, recall, f1-score)
    metrics_cols = ['precision', 'recall', 'f1-score', 'support']
    metrics_df = metrics_df[metrics_cols]
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        metrics_df[['precision', 'recall', 'f1-score']],
        annot=True,
        fmt='.3f',
        cmap='Blues',
        vmin=0,
        vmax=1
    )
    plt.title('Per-Class Metrics')
    plt.tight_layout()
    
    # Save plot
    cr_path = output_path / "classification_report.png"
    plt.savefig(cr_path, dpi=150, bbox_inches='tight')
    plt.close()
    


def plot_metrics_comparison(
    results: Dict[str, Any], 
    output_path: Path
) -> None:
    """
    Plot metrics comparison bar chart.
    
    Args:
        results: Evaluation results dictionary
        output_path: Output directory path
    """
    # Extract macro and weighted metrics
    metrics = {
        'Accuracy': results['accuracy'],
        'Precision (Macro)': results['macro_avg']['precision'],
        'Recall (Macro)': results['macro_avg']['recall'],
        'F1-Score (Macro)': results['macro_f1'],
        'F1-Score (Weighted)': results['weighted_avg']['f1-score']
    }
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(metrics)), list(metrics.values()), align='center')
    plt.xticks(range(len(metrics)), list(metrics.keys()), rotation=45, ha='right')
    plt.ylabel('Score')
    plt.title('Model Performance Metrics')
    plt.ylim(0, 1)
    
    # Add value labels on top of bars
    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    
    # Save plot
    metrics_path = output_path / "metrics_comparison.png"
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    plt.close()



def generate_visualizations(
    results: Dict[str, Any], 
    output_path: Path, 
    label_info: Dict[str, Any]
) -> None:
    """
    Generate and save all visualizations for zero-shot evaluation results.
    
    Args:
        results: Evaluation results dictionary
        output_path: Output directory path
        label_info: Label information dictionary
    """
    logger.info("Generating visualizations...")
    
    # Extract true and predicted labels if available
    # Note: We don't store raw labels in results, so we can't generate confusion matrix
    # We'll focus on other visualizations for now
    
    # Plot classification report
    plot_classification_report(results, output_path)
    
    # Plot metrics comparison
    plot_metrics_comparison(results, output_path)
    
    logger.info(f"Generated visualizations saved to {output_path}")


def convert_labels_to_ids(df: pd.DataFrame, label_info: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    """
    Convert ground truth labels to integer IDs.
    
    Args:
        df: DataFrame with 'label' column containing label names
        label_info: Dictionary with 'label2id' mapping
    
    Returns:
        Tuple of (y_true_ids, valid_mask) where valid_mask indicates valid labels
    """
    label2id = label_info['label2id']
    y_true_ids = []
    valid_mask = []
    
    for label in df['label']:
        if label in label2id:
            y_true_ids.append(label2id[label])
            valid_mask.append(True)
        else:
            logger.warning(f"Label '{label}' not found in label mapping, skipping")
            y_true_ids.append(-1)  # Invalid label
            valid_mask.append(False)
    
    return y_true_ids, valid_mask


def evaluate_gpt_zero_shot(
    test_df: pd.DataFrame,
    label_info: Dict[str, Any],
    prompt_template: str,
    model: str = "gpt-4o-mini",
    output_dir: str = "experiments/E900_gpt4o_zero_shot"
) -> Dict[str, Any]:
    """
    Evaluate GPT zero-shot classification.
    
    Args:
        test_df: Test DataFrame with text, label, language columns
        label_info: Label information dictionary
        prompt_template: Prompt template string
        model: GPT model name (default: "gpt-4o-mini")
        output_dir: Output directory for results
    
    Returns:
        Dictionary containing evaluation results and metrics
    """
    logger.info("="*60)
    logger.info("GPT Zero-Shot Evaluation")
    logger.info("="*60)
    logger.info(f"Model: {model}")
    logger.info(f"Test examples: {len(test_df)}")
    
    # Build all prompts
    prompts = []
    for _, row in test_df.iterrows():
        prompt = render_prompt(
            prompt_template,
            row['text'],
            row['language'],
            label_info
        )
        prompts.append(prompt)
    
    logger.info(f"Built {len(prompts)} prompts")
    
    # Call GPT API
    logger.info("Calling GPT API...")
    responses = gpt_zero_shot_predict(prompts, model=model)
    logger.info(f"Received {len(responses)} responses")
    
    # Parse responses to label IDs
    y_pred_ids = []
    parsing_failures = []
    
    for i, response in enumerate(responses):
        label_id = parse_label_id(response)
        if label_id is not None:
            y_pred_ids.append(label_id)
        else:
            # Parsing failed - use majority class as fallback
            # Find majority class from valid labels
            valid_labels = [l for l in test_df['label'] if l in label_info['label2id']]
            if valid_labels:
                majority_label = pd.Series(valid_labels).mode()[0]
                fallback_id = label_info['label2id'][majority_label]
                y_pred_ids.append(fallback_id)
                parsing_failures.append({
                    'index': i,
                    'response': response,
                    'fallback_label_id': fallback_id
                })
                logger.warning(f"Failed to parse response '{response}', using fallback label {fallback_id}")
            else:
                # No valid labels, use 0 as last resort
                y_pred_ids.append(0)
                parsing_failures.append({
                    'index': i,
                    'response': response,
                    'fallback_label_id': 0
                })
                logger.error(f"Failed to parse response '{response}' and no valid labels, using 0")
    
    # Convert ground truth labels to IDs
    y_true_ids, valid_mask = convert_labels_to_ids(test_df, label_info)
    
    # Filter to only valid examples
    valid_indices = [i for i, valid in enumerate(valid_mask) if valid]
    y_true_filtered = [y_true_ids[i] for i in valid_indices]
    y_pred_filtered = [y_pred_ids[i] for i in valid_indices]
    
    logger.info(f"Valid examples: {len(y_true_filtered)}/{len(test_df)}")
    logger.info(f"Parsing failures: {len(parsing_failures)}")
    
    # Get label names for reporting
    id2label = label_info['id2label']
    label_names = [id2label[str(i)] for i in sorted([int(k) for k in id2label.keys()])]
    
    # Compute metrics
    logger.info("Computing metrics...")
    classification_results = classification_report_with_macro_f1(
        y_true_filtered,
        y_pred_filtered,
        label_names=label_names
    )
    
    # Prepare results
    results = {
        'model': model,
        'model_type': 'gpt',
        'num_examples': len(test_df),
        'num_valid_examples': len(y_true_filtered),
        'num_parsing_failures': len(parsing_failures),
        'parsing_failures': parsing_failures[:10],  # Store first 10 for inspection
        'macro_f1': classification_results['macro_f1'],
        'accuracy': classification_results['accuracy'],
        'per_class_metrics': classification_results['per_class'],
        'macro_avg': classification_results['macro_avg'],
        'weighted_avg': classification_results['weighted_avg']
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved results to {metrics_path}")
    logger.info(f"Macro-F1: {results['macro_f1']:.4f}")
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    
    # Generate and save visualizations
    generate_visualizations(results, output_path, label_info)
    
    return results


def evaluate_deepseek_zero_shot(
    test_df: pd.DataFrame,
    label_info: Dict[str, Any],
    prompt_template: str,
    model: str = "deepseek-chat",
    output_dir: str = "experiments/E901_deepseek_zero_shot"
) -> Dict[str, Any]:
    """
    Evaluate DeepSeek zero-shot classification.
    
    Args:
        test_df: Test DataFrame with text, label, language columns
        label_info: Label information dictionary
        prompt_template: Prompt template string
        model: DeepSeek model name (default: "deepseek-chat")
        output_dir: Output directory for results
    
    Returns:
        Dictionary containing evaluation results and metrics
    """
    logger.info("="*60)
    logger.info("DeepSeek Zero-Shot Evaluation")
    logger.info("="*60)
    logger.info(f"Model: {model}")
    logger.info(f"Test examples: {len(test_df)}")
    
    # Build all prompts
    prompts = []
    for _, row in test_df.iterrows():
        prompt = render_prompt(
            prompt_template,
            row['text'],
            row['language'],
            label_info
        )
        prompts.append(prompt)
    
    logger.info(f"Built {len(prompts)} prompts")
    
    # Call DeepSeek API
    logger.info("Calling DeepSeek API...")
    responses = deepseek_zero_shot_predict(prompts, model=model)
    logger.info(f"Received {len(responses)} responses")
    
    # Parse responses to label IDs
    y_pred_ids = []
    parsing_failures = []
    
    for i, response in enumerate(responses):
        label_id = parse_label_id(response)
        if label_id is not None:
            y_pred_ids.append(label_id)
        else:
            # Parsing failed - use majority class as fallback
            valid_labels = [l for l in test_df['label'] if l in label_info['label2id']]
            if valid_labels:
                majority_label = pd.Series(valid_labels).mode()[0]
                fallback_id = label_info['label2id'][majority_label]
                y_pred_ids.append(fallback_id)
                parsing_failures.append({
                    'index': i,
                    'response': response,
                    'fallback_label_id': fallback_id
                })
                logger.warning(f"Failed to parse response '{response}', using fallback label {fallback_id}")
            else:
                y_pred_ids.append(0)
                parsing_failures.append({
                    'index': i,
                    'response': response,
                    'fallback_label_id': 0
                })
                logger.error(f"Failed to parse response '{response}' and no valid labels, using 0")
    
    # Convert ground truth labels to IDs
    y_true_ids, valid_mask = convert_labels_to_ids(test_df, label_info)
    
    # Filter to only valid examples
    valid_indices = [i for i, valid in enumerate(valid_mask) if valid]
    y_true_filtered = [y_true_ids[i] for i in valid_indices]
    y_pred_filtered = [y_pred_ids[i] for i in valid_indices]
    
    logger.info(f"Valid examples: {len(y_true_filtered)}/{len(test_df)}")
    logger.info(f"Parsing failures: {len(parsing_failures)}")
    
    # Get label names for reporting
    id2label = label_info['id2label']
    label_names = [id2label[str(i)] for i in sorted([int(k) for k in id2label.keys()])]
    
    # Compute metrics
    logger.info("Computing metrics...")
    classification_results = classification_report_with_macro_f1(
        y_true_filtered,
        y_pred_filtered,
        label_names=label_names
    )
    
    # Prepare results
    results = {
        'model': model,
        'model_type': 'deepseek',
        'num_examples': len(test_df),
        'num_valid_examples': len(y_true_filtered),
        'num_parsing_failures': len(parsing_failures),
        'parsing_failures': parsing_failures[:10],  # Store first 10 for inspection
        'macro_f1': classification_results['macro_f1'],
        'accuracy': classification_results['accuracy'],
        'per_class_metrics': classification_results['per_class'],
        'macro_avg': classification_results['macro_avg'],
        'weighted_avg': classification_results['weighted_avg']
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved results to {metrics_path}")
    logger.info(f"Macro-F1: {results['macro_f1']:.4f}")
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    
    # Generate and save visualizations
    generate_visualizations(results, output_path, label_info)
    
    return results


def main():
    """Main entry point for zero-shot LLM evaluation."""
    parser = argparse.ArgumentParser(
        description="Zero-shot LLM evaluation for orthopedic diagnostic classification"
    )
    parser.add_argument(
        "--engine",
        required=True,
        choices=["gpt4o", "deepseek"],
        help="LLM engine to evaluate"
    )
    parser.add_argument(
        "--test",
        default="preprocess_jsonl/english_data.jsonl",
        help="Path to test data file (CSV or JSONL)"
    )
    parser.add_argument(
        "--prompt",
        default="src/llm/prompts/zero_shot.txt",
        help="Path to prompt template file"
    )
    parser.add_argument(
        "--labels",
        default="data/processed/labels.json",
        help="Path to labels.json"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: experiments/E900_gpt4o_zero_shot or E901_deepseek_zero_shot)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override (default: gpt-4o-mini for GPT, deepseek-chat for DeepSeek)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to configuration file (optional)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set default output directory based on engine
    if args.out is None:
        if args.engine == "gpt4o":
            output_dir = "experiments/E900_gpt4o_zero_shot"
        else:
            output_dir = "experiments/E901_deepseek_zero_shot"
    else:
        output_dir = args.out
    
    # Load data
    logger.info("Loading test data and labels...")
    test_df = load_test_data(args.test)
    label_info = load_label_mapping(args.labels)
    prompt_template = load_prompt_template(args.prompt)
    
    # Route to appropriate evaluation function
    if args.engine == "gpt4o":
        model = args.model or "gpt-4o-mini"
        evaluate_gpt_zero_shot(
            test_df,
            label_info,
            prompt_template,
            model=model,
            output_dir=output_dir
        )
    elif args.engine == "deepseek":
        model = args.model or "deepseek-chat"
        evaluate_deepseek_zero_shot(
            test_df,
            label_info,
            prompt_template,
            model=model,
            output_dir=output_dir
        )
    else:
        raise ValueError(f"Unknown engine: {args.engine}")
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
