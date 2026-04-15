"""
Evaluate all supervised models on test set.

Scans all experiment directories and evaluates each trained model.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import glob

import pandas as pd

from src.eval.run_single_supervised import evaluate_single_model

logger = logging.getLogger(__name__)


def find_latest_experiment(experiment_root: str) -> Optional[str]:
    """Find the most recent experiment ID in the experiment root directory."""
    if not os.path.exists(experiment_root):
        return None
    
    experiment_dirs = [
        d for d in os.listdir(experiment_root)
        if os.path.isdir(os.path.join(experiment_root, d)) and d.startswith('E')
    ]
    
    if not experiment_dirs:
        return None
    
    # Sort by creation time (most recent first)
    experiment_dirs_with_time = [
        (d, os.path.getctime(os.path.join(experiment_root, d)))
        for d in experiment_dirs
    ]
    experiment_dirs_with_time.sort(key=lambda x: x[1], reverse=True)
    
    return experiment_dirs_with_time[0][0]


def discover_all_experiments(experiments_root: str) -> List[Dict[str, Any]]:
    """
    Discover all experiment runs across all backbones.
    
    Returns:
        List of dicts with keys: backbone, experiment_id, languages
    """
    experiments = []
    
    backbones = ['distilbert', 'indicbert', 'indicbert_hpa', 'xlmroberta', 'mdeberta']
    
    for backbone in backbones:
        backbone_dir = os.path.join(experiments_root, backbone)
        if not os.path.exists(backbone_dir):
            logger.debug(f"Backbone directory not found: {backbone_dir}")
            continue
        
        # Find all experiment IDs
        experiment_dirs = [
            d for d in os.listdir(backbone_dir)
            if os.path.isdir(os.path.join(backbone_dir, d)) and d.startswith('E')
        ]
        
        for exp_id in experiment_dirs:
            exp_dir = os.path.join(backbone_dir, exp_id)
            
            # Check which languages exist
            languages = []
            for lang in ['en', 'hi', 'pa']:
                lang_dir = os.path.join(exp_dir, lang)
                if os.path.exists(lang_dir):
                    # Check if checkpoint exists
                    checkpoint = os.path.join(lang_dir, 'best.ckpt')
                    model_dir = os.path.join(lang_dir, 'model')
                    if os.path.exists(checkpoint) or os.path.exists(model_dir):
                        languages.append(lang)
            
            if languages:
                experiments.append({
                    'backbone': backbone,
                    'experiment_id': exp_id,
                    'languages': languages,
                    'experiment_dir': exp_dir
                })
    
    return experiments


def main():
    """Main function for evaluating all supervised models."""
    parser = argparse.ArgumentParser(
        description='Evaluate all supervised models on test set',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--experiments_root',
        type=str,
        default='experiments',
        help='Root directory containing experiment subdirectories (default: experiments)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='reports/eval_supervised',
        help='Output directory for evaluation results (default: reports/eval_supervised)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Data split to evaluate on (default: test)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get project root
    project_root = Path(__file__).resolve().parents[2]
    experiments_root = project_root / args.experiments_root
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Scanning experiments in: {experiments_root}")
    logger.info(f"Output directory: {output_dir}")
    
    # Discover all experiments
    all_experiments = discover_all_experiments(str(experiments_root))
    
    if not all_experiments:
        logger.warning("No experiments found. Make sure you have trained models first.")
        return
    
    logger.info(f"Found {len(all_experiments)} experiment runs")
    
    # Evaluate each experiment
    all_results = []
    
    for exp_info in all_experiments:
        backbone = exp_info['backbone']
        exp_id = exp_info['experiment_id']
        languages = exp_info['languages']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {backbone} / {exp_id}")
        logger.info(f"Languages: {', '.join(languages)}")
        logger.info(f"{'='*60}")
        
        for lang in languages:
            model_root = os.path.join(exp_info['experiment_dir'], lang)
            lang_output_dir = output_dir / backbone / exp_id / lang
            lang_output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"\nEvaluating {backbone} / {exp_id} / {lang}")
            
            try:
                results = evaluate_single_model(
                    model_root=model_root,
                    output_dir=str(lang_output_dir),
                    split=args.split
                )
                
                if results:
                    results['backbone'] = backbone
                    results['experiment_id'] = exp_id
                    all_results.append(results)
                    logger.info(f"✓ Successfully evaluated {backbone} / {exp_id} / {lang}")
                else:
                    logger.warning(f"✗ No results for {backbone} / {exp_id} / {lang}")
                    
            except Exception as e:
                logger.error(f"✗ Failed to evaluate {backbone} / {exp_id} / {lang}: {e}")
                continue
    
    # Save combined summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_csv = output_dir / f'all_results_{args.split}.csv'
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"\nSaved combined summary to {summary_csv}")
        
        summary_json = output_dir / f'all_results_{args.split}.json'
        with open(summary_json, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved combined JSON to {summary_json}")
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)
        for result in all_results:
            metrics = result.get('metrics', {})
            logger.info(
                f"{result['backbone']} / {result['experiment_id']} / {result['language']}: "
                f"Acc={metrics.get('accuracy', 0.0):.4f}, "
                f"F1_macro={metrics.get('f1_macro', 0.0):.4f}"
            )
        logger.info("="*80)
    else:
        logger.warning("No results to summarize.")


if __name__ == '__main__':
    main()

