import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import logging
import os
from typing import Dict, Any, List
from pathlib import Path
import sys

# Add the parent directory to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineEvaluator:
    """
    Pipeline Evaluator - Computes metrics and generates reports for pipeline results.
    """
    
    def __init__(self):
        """
        Initialize the Pipeline Evaluator.
        """
        logger.info("Pipeline Evaluator initialized")
        
        # Label mapping to standardize true labels across languages
        self.label_mapping = {
            # Spinal disorders
            'Spinal disorders': 'spinal_disorder',
            'रीढ़ से संबंधित विकार': 'spinal_disorder',
            'ਰੀੜ੍ਹ ਦੀ ਹੱਡੀ ਦੇ ਵਿਕਾਰ': 'spinal_disorder',
            'कमर ਨਾਲ ਸਬੰਧਤ ਵਿਕਾਰ': 'spinal_disorder',
            
            # Fracture/Bone disorders
            'Fracture': 'fracture',
            'Bone-related disorders': 'fracture',
            'हड्डी संबंधित विकार': 'fracture',
            'ਹੱਡੀਆਂ ਨਾਲ ਸਬੰਧਤ ਵਿਕਾਰ': 'fracture',
            
            # Musculoskeletal disorders
            'Musculoskeletal disorders': 'musculoskeletal_disorder',
            'मस्कुलोस्केलेटल विकार': 'musculoskeletal_disorder',
            'ਮਸੂਕਲੋਸਕੇਲਟਲ ਵਿਕਾਰ': 'musculoskeletal_disorder',
            'Hip-related disorders': 'musculoskeletal_disorder',
            'कूल्हे से संबंधित विकार': 'musculoskeletal_disorder',
            
            # Other/Unknown
            'Other': 'other',
            'अन्य': 'other',
            'हੋਰ': 'other',
            'Unknown': 'other',
            'अगਿਆਤ': 'other'
        }
        
    def _standardize_labels(self, labels):
        """
        Standardize labels using the predefined mapping.
        
        Args:
            labels: List or Series of labels to standardize
            
        Returns:
            Standardized labels
        """
        return labels.map(lambda x: self.label_mapping.get(x, 'other'))
    
    def load_pipeline_results(self, results_path: str) -> pd.DataFrame:
        """
        Load pipeline results from CSV file.
        
        Args:
            results_path: Path to pipeline predictions CSV file
            
        Returns:
            DataFrame containing pipeline results
        """
        logger.info(f"Loading pipeline results from: {results_path}")
        results_df = pd.read_csv(results_path)
        logger.info(f"Loaded {len(results_df)} samples")
        return results_df
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray = None) -> Dict[str, Any]:
        """
        Compute various evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Predicted probabilities (optional)
            
        Returns:
            Dictionary containing computed metrics
        """
        # Convert to pandas Series for standardization
        y_true_series = pd.Series(y_true)
        
        # Standardize ONLY true labels (predicted labels are already standardized by the model)
        y_true_std = self._standardize_labels(y_true_series).values
        y_pred_std = np.array(y_pred)  # Use predicted labels as-is
        
        metrics = {
            'accuracy': accuracy_score(y_true_std, y_pred_std),
            'f1_macro': f1_score(y_true_std, y_pred_std, average='macro'),
            'balanced_accuracy': balanced_accuracy_score(y_true_std, y_pred_std),
            'confusion_matrix': confusion_matrix(y_true_std, y_pred_std).tolist()
        }
        
        # Compute AUROC and AUPRC if scores are provided
        if y_scores is not None:
            try:
                metrics['auroc'] = roc_auc_score(y_true, y_scores, multi_class='ovr', average='macro')
            except ValueError as e:
                logger.warning(f"Could not compute AUROC: {e}")
                metrics['auroc'] = None
            
            try:
                metrics['auprc'] = average_precision_score(y_true, y_scores, average='macro')
            except ValueError as e:
                logger.warning(f"Could not compute AUPRC: {e}")
                metrics['auprc'] = None
        
        return metrics
    
    def compute_hil_metrics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute HIL-specific metrics like coverage, escalations, etc.
        
        Args:
            results_df: DataFrame containing pipeline results
            
        Returns:
            Dictionary containing HIL metrics
        """
        total_samples = len(results_df)
        
        # Calculate coverage (% AUTO_ACCEPT)
        auto_accept_samples = results_df[results_df['hil_decision'] == 'AUTO_ACCEPT']
        coverage = len(auto_accept_samples) / total_samples if total_samples > 0 else 0
        
        # Calculate escalations (% REQUIRE_REVIEW)
        escalations = 1 - coverage
        
        # Calculate accuracy on AUTO_ACCEPT subset
        auto_accept_accuracy = 0.0
        if len(auto_accept_samples) > 0:
            # Standardize ONLY true labels (predicted labels are already standardized)
            y_true_std = self._standardize_labels(auto_accept_samples['true_label'])
            y_pred_std = auto_accept_samples['predicted_label']  # Use predicted labels as-is
            auto_accept_accuracy = accuracy_score(y_true_std, y_pred_std)
        
        # Calculate "system accuracy after review" (simulate review by replacing escalated predictions with true_label)
        reviewed_predictions = results_df['predicted_label'].copy()
        review_mask = results_df['hil_decision'] == 'REQUIRE_REVIEW'
        
        # For escalated cases, we need to standardize the true labels before replacing
        true_labels_standardized = self._standardize_labels(results_df['true_label'])
        reviewed_predictions[review_mask] = true_labels_standardized[review_mask]
        
        # Standardize ONLY true labels (predicted labels are already standardized)
        y_true_std = true_labels_standardized
        y_pred_std = reviewed_predictions  # Use reviewed predictions as-is
        system_accuracy_after_review = accuracy_score(y_true_std, y_pred_std)
        
        # Calculate per-language metrics
        language_metrics = {}
        for lang in results_df['language'].unique():
            lang_df = results_df[results_df['language'] == lang]
            lang_total = len(lang_df)
            
            lang_auto_accept = len(lang_df[lang_df['hil_decision'] == 'AUTO_ACCEPT'])
            lang_coverage = lang_auto_accept / lang_total if lang_total > 0 else 0
            
            language_metrics[lang] = {
                'total_samples': lang_total,
                'coverage': lang_coverage,
                'escalations': 1 - lang_coverage
            }
        
        return {
            'coverage': coverage,
            'escalations': escalations,
            'auto_accept_accuracy': auto_accept_accuracy,
            'system_accuracy_after_review': system_accuracy_after_review,
            'language_metrics': language_metrics
        }
    
    def evaluate_pipeline(self, results_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Evaluate pipeline results and generate reports.
        
        Args:
            results_path: Path to pipeline predictions CSV file
            output_dir: Directory to save evaluation reports (optional)
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Evaluating pipeline results from: {results_path}")
        
        # Load results
        results_df = self.load_pipeline_results(results_path)
        
        # Compute overall metrics
        overall_metrics = self.compute_metrics(
            results_df['true_label'],
            results_df['predicted_label']
        )
        
        # Compute HIL metrics
        hil_metrics = self.compute_hil_metrics(results_df)
        
        # Compute metrics by language
        metrics_by_language = {}
        for lang in results_df['language'].unique():
            lang_df = results_df[results_df['language'] == lang]
            metrics_by_language[lang] = self.compute_metrics(
                lang_df['true_label'],
                lang_df['predicted_label']
            )
        
        # Compile evaluation results
        evaluation_results = {
            'overall': {
                'metrics': overall_metrics,
                'hil_metrics': hil_metrics,
                'sample_count': len(results_df)
            },
            'by_language': metrics_by_language
        }
        
        # Save report if output directory is provided
        if output_dir:
            self._save_evaluation_report(evaluation_results, results_path, output_dir)
        
        logger.info("Pipeline evaluation completed")
        return evaluation_results
    
    def evaluate_ablation_studies(self, ablation_results: Dict[str, pd.DataFrame], output_dir: str = None) -> Dict[str, Any]:
        """
        Evaluate ablation studies and generate comparative reports.
        
        Args:
            ablation_results: Dictionary mapping ablation modes to their results DataFrames
            output_dir: Directory to save ablation reports (optional)
            
        Returns:
            Dictionary containing ablation evaluation results
        """
        logger.info("Evaluating ablation studies")
        
        ablation_evaluations = {}
        
        for mode, results_df in ablation_results.items():
            logger.info(f"Evaluating ablation mode: {mode}")
            
            # Compute metrics for this ablation mode
            metrics = self.compute_metrics(
                results_df['true_label'],
                results_df['predicted_label']
            )
            
            # Compute HIL metrics if applicable
            hil_metrics = self.compute_hil_metrics(results_df)
            
            ablation_evaluations[mode] = {
                'metrics': metrics,
                'hil_metrics': hil_metrics,
                'sample_count': len(results_df)
            }
        
        # Generate ablation summary
        ablation_summary = self._generate_ablation_summary(ablation_evaluations)
        
        # Save ablation report if output directory is provided
        if output_dir:
            self._save_ablation_report(ablation_evaluations, ablation_summary, output_dir)
        
        logger.info("Ablation studies evaluation completed")
        return {
            'ablation_evaluations': ablation_evaluations,
            'ablation_summary': ablation_summary
        }
    
    def _generate_ablation_summary(self, ablation_evaluations: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate ablation summary DataFrame.
        
        Args:
            ablation_evaluations: Dictionary containing ablation evaluations
            
        Returns:
            DataFrame containing ablation summary
        """
        # Create summary table
        summary_data = []
        
        for mode, eval_result in ablation_evaluations.items():
            summary_data.append({
                'ablation_mode': mode,
                'accuracy': eval_result['metrics']['accuracy'],
                'f1_macro': eval_result['metrics']['f1_macro'],
                'balanced_accuracy': eval_result['metrics']['balanced_accuracy'],
                'coverage': eval_result['hil_metrics']['coverage'],
                'escalations': eval_result['hil_metrics']['escalations'],
                'auto_accept_accuracy': eval_result['hil_metrics']['auto_accept_accuracy'],
                'system_accuracy_after_review': eval_result['hil_metrics']['system_accuracy_after_review'],
                'sample_count': eval_result['sample_count']
            })
        
        return pd.DataFrame(summary_data)
    
    def _save_evaluation_report(self, evaluation_results: Dict[str, Any], results_path: str, output_dir: str):
        """
        Save evaluation report to output directory.
        
        Args:
            evaluation_results: Dictionary containing evaluation results
            results_path: Path to pipeline results file
            output_dir: Directory to save report
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate report filename
        report_filename = f"evaluation_report_{os.path.basename(results_path).replace('.csv', '')}.md"
        report_path = os.path.join(output_dir, report_filename)
        
        logger.info(f"Saving evaluation report to: {report_path}")
        
        # Write report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Pipeline Evaluation Report\n\n")
            f.write(f"## Results File: {os.path.basename(results_path)}\n\n")
            
            # Overall Metrics
            f.write("## Overall Metrics\n\n")
            overall = evaluation_results['overall']
            f.write("### Model Performance\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Accuracy | {overall['metrics']['accuracy']:.4f} |\n")
            f.write(f"| F1 Macro | {overall['metrics']['f1_macro']:.4f} |\n")
            f.write(f"| Balanced Accuracy | {overall['metrics']['balanced_accuracy']:.4f} |\n")
            
            if 'auroc' in overall['metrics'] and overall['metrics']['auroc'] is not None:
                f.write(f"| AUROC | {overall['metrics']['auroc']:.4f} |\n")
            
            if 'auprc' in overall['metrics'] and overall['metrics']['auprc'] is not None:
                f.write(f"| AUPRC | {overall['metrics']['auprc']:.4f} |\n")
            
            # HIL Metrics
            f.write("\n### HIL Performance\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Coverage (AUTO_ACCEPT %) | {overall['hil_metrics']['coverage']:.2%} |\n")
            f.write(f"| Escalations (REQUIRE_REVIEW %) | {overall['hil_metrics']['escalations']:.2%} |\n")
            f.write(f"| Accuracy on AUTO_ACCEPT | {overall['hil_metrics']['auto_accept_accuracy']:.4f} |\n")
            f.write(f"| System Accuracy After Review | {overall['hil_metrics']['system_accuracy_after_review']:.4f} |\n")
            
            # By Language Metrics
            f.write("\n## Metrics by Language\n\n")
            for lang, metrics in evaluation_results['by_language'].items():
                f.write(f"### {lang.upper()}\n\n")
                f.write(f"| Metric | Value |\n")
                f.write(f"|--------|-------|\n")
                f.write(f"| Accuracy | {metrics['accuracy']:.4f} |\n")
                f.write(f"| F1 Macro | {metrics['f1_macro']:.4f} |\n")
                f.write(f"| Balanced Accuracy | {metrics['balanced_accuracy']:.4f} |\n")
                
                # Add language-specific HIL metrics
                if lang in overall['hil_metrics']['language_metrics']:
                    lang_hil = overall['hil_metrics']['language_metrics'][lang]
                    f.write(f"| Coverage | {lang_hil['coverage']:.2%} |\n")
                    f.write(f"| Escalations | {lang_hil['escalations']:.2%} |\n")
                
                f.write("\n")
    
    def _save_ablation_report(self, ablation_evaluations: Dict[str, Any], ablation_summary: pd.DataFrame, output_dir: str):
        """
        Save ablation report to output directory.
        
        Args:
            ablation_evaluations: Dictionary containing ablation evaluations
            ablation_summary: DataFrame containing ablation summary
            output_dir: Directory to save report
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary CSV
        summary_csv_path = os.path.join(output_dir, 'ablation_summary.csv')
        ablation_summary.to_csv(summary_csv_path, index=False)
        logger.info(f"Saved ablation summary to: {summary_csv_path}")
        
        # Save detailed report
        report_path = os.path.join(output_dir, 'ablation_report.md')
        logger.info(f"Saving ablation report to: {report_path}")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Ablation Study Report\n\n")
            
            # Write summary table
            f.write("## Ablation Summary\n\n")
            f.write(ablation_summary.to_markdown(index=False))
            f.write("\n\n")
            
            # Write detailed results for each ablation mode
            f.write("## Detailed Results by Ablation Mode\n\n")
            for mode, eval_result in ablation_evaluations.items():
                f.write(f"### Ablation Mode: {mode}\n\n")
                f.write("#### Model Performance\n\n")
                f.write(f"| Metric | Value |\n")
                f.write(f"|--------|-------|\n")
                f.write(f"| Accuracy | {eval_result['metrics']['accuracy']:.4f} |\n")
                f.write(f"| F1 Macro | {eval_result['metrics']['f1_macro']:.4f} |\n")
                f.write(f"| Balanced Accuracy | {eval_result['metrics']['balanced_accuracy']:.4f} |\n")
                
                f.write("\n#### HIL Performance\n\n")
                f.write(f"| Metric | Value |\n")
                f.write(f"|--------|-------|\n")
                f.write(f"| Coverage (AUTO_ACCEPT %) | {eval_result['hil_metrics']['coverage']:.2%} |\n")
                f.write(f"| Escalations (REQUIRE_REVIEW %) | {eval_result['hil_metrics']['escalations']:.2%} |\n")
                f.write(f"| Accuracy on AUTO_ACCEPT | {eval_result['hil_metrics']['auto_accept_accuracy']:.4f} |\n")
                f.write(f"| System Accuracy After Review | {eval_result['hil_metrics']['system_accuracy_after_review']:.4f} |\n")
                f.write("\n")

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Pipeline Results')
    
    # Input arguments
    parser.add_argument('--results_path', type=str, required=False,
                        help='Path to pipeline predictions CSV file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save evaluation reports')
    parser.add_argument('--ablation_study', action='store_true',
                        help='Run ablation study comparison')
    parser.add_argument('--ablation_files', type=str, nargs='+',
                        help='List of ablation results files for comparison')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = PipelineEvaluator()
    
    if args.ablation_study:
        # Run ablation study
        if not args.ablation_files:
            raise ValueError("--ablation_files is required when --ablation_study is specified")
        
        # Load all ablation results
        ablation_results = {}
        for file_path in args.ablation_files:
            mode = os.path.basename(file_path).replace('pipeline_predictions_', '').replace('.csv', '')
            ablation_results[mode] = evaluator.load_pipeline_results(file_path)
        
        # Evaluate ablation studies
        evaluator.evaluate_ablation_studies(ablation_results, args.output_dir)
    else:
        # Evaluate single pipeline results
        evaluator.evaluate_pipeline(args.results_path, args.output_dir)
