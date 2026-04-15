import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
import os
import json

logger = logging.getLogger(__name__)

def analyze_data_quality(df: pd.DataFrame, language: str) -> Dict[str, Any]:
    """Analyze data quality for a given language dataset."""
    analysis = {
        'language': language,
        'total_records': len(df),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'text_statistics': {
            'min_length': df['text'].str.len().min(),
            'max_length': df['text'].str.len().max(),
            'mean_length': df['text'].str.len().mean(),
            'empty_texts': (df['text'].str.len() == 0).sum()
        },
        'label_distribution': df['label'].value_counts().to_dict(),
        'unique_labels': len(df['label'].unique())
    }
    
    return analysis

def generate_language_specific_datasets(processed_dir: str, output_dir: str):
    """Generate language-specific datasets for separate training."""
    ensure_dir(output_dir)
    
    # Load processed data
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_path = os.path.join(processed_dir, f"{split}.csv")
        if not os.path.exists(split_path):
            logger.warning(f"Split file not found: {split_path}")
            continue
            
        df = pd.read_csv(split_path)
        
        # Create language-specific files
        for language in df['language'].unique():
            lang_df = df[df['language'] == language]
            lang_output_path = os.path.join(output_dir, f"{language.lower()}_{split}.csv")
            lang_df.to_csv(lang_output_path, index=False)
            logger.info(f"Saved {language} {split} data: {len(lang_df)} records")
    
    # Copy labels file
    labels_path = os.path.join(processed_dir, "labels.json")
    if os.path.exists(labels_path):
        import shutil
        shutil.copy2(labels_path, os.path.join(output_dir, "labels.json"))

def get_dataset_statistics(processed_dir: str) -> Dict[str, Any]:
    """Get comprehensive statistics for the processed dataset."""
    stats_path = os.path.join(processed_dir, "statistics.json")
    
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Calculate statistics if not available
    splits = {}
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(processed_dir, f"{split}.csv")
        if os.path.exists(split_path):
            splits[split] = pd.read_csv(split_path)
    
    if not splits:
        raise ValueError("No processed data found")
    
    # Calculate basic statistics
    stats = {}
    for split_name, split_df in splits.items():
        stats[split_name] = {
            'total_records': len(split_df),
            'language_distribution': split_df['language'].value_counts().to_dict(),
            'label_distribution': split_df['label'].value_counts().to_dict()
        }
    
    return stats

def validate_processed_data(processed_dir: str) -> bool:
    """Validate that processed data meets requirements."""
    required_files = ['train.csv', 'val.csv', 'test.csv', 'labels.json']
    
    for file in required_files:
        file_path = os.path.join(processed_dir, file)
        if not os.path.exists(file_path):
            logger.error(f"Required file missing: {file_path}")
            return False
    
    # Check that files are not empty
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(processed_dir, f"{split}.csv")
        df = pd.read_csv(split_path)
        if df.empty:
            logger.error(f"Split file is empty: {split_path}")
            return False
    
    logger.info("Processed data validation passed")
    return True