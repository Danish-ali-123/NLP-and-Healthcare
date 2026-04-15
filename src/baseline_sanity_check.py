#!/usr/bin/env python3
"""
Sanity check script for baseline experiments
- Verify label-to-id mapping consistency
- Verify train/val/test splits have no overlap
- Verify metrics are computed correctly
- Run random and majority baselines
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import os
import json
import argparse
import hashlib


def hash_text(text):
    """Create a hash of text for checking overlap"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def check_data_overlap(train_df, val_df, test_df):
    """Check if there's any overlap between train, val, and test sets"""
    print("Checking for data overlap...")
    
    # Create hashes of texts
    train_hashes = set(train_df['text_input'].apply(hash_text))
    val_hashes = set(val_df['text_input'].apply(hash_text))
    test_hashes = set(test_df['text_input'].apply(hash_text))
    
    # Check overlaps
    train_val_overlap = train_hashes.intersection(val_hashes)
    train_test_overlap = train_hashes.intersection(test_hashes)
    val_test_overlap = val_hashes.intersection(test_hashes)
    
    overlap_results = {
        'train_val_overlap': len(train_val_overlap),
        'train_test_overlap': len(train_test_overlap),
        'val_test_overlap': len(val_test_overlap),
        'has_overlap': len(train_val_overlap) > 0 or len(train_test_overlap) > 0 or len(val_test_overlap) > 0
    }
    
    print(f"Train-Validation overlap: {len(train_val_overlap)}")
    print(f"Train-Test overlap: {len(train_test_overlap)}")
    print(f"Validation-Test overlap: {len(val_test_overlap)}")
    
    if overlap_results['has_overlap']:
        print("WARNING: Data overlap found between splits!")
    else:
        print("✓ No data overlap between splits")
    
    return overlap_results


def check_label_consistency(train_df, val_df, test_df, label_col):
    """Check if labels are consistent across splits"""
    print("\nChecking label consistency...")
    
    train_labels = set(train_df[label_col].unique())
    val_labels = set(val_df[label_col].unique())
    test_labels = set(test_df[label_col].unique())
    
    # Check if val/test labels are subset of train labels
    val_labels_in_train = val_labels.issubset(train_labels)
    test_labels_in_train = test_labels.issubset(train_labels)
    
    consistency_results = {
        'train_labels': sorted(train_labels),
        'val_labels': sorted(val_labels),
        'test_labels': sorted(test_labels),
        'val_labels_in_train': val_labels_in_train,
        'test_labels_in_train': test_labels_in_train,
        'all_labels_consistent': val_labels_in_train and test_labels_in_train
    }
    
    print(f"Train labels ({len(train_labels)}): {consistency_results['train_labels']}")
    print(f"Val labels ({len(val_labels)}): {consistency_results['val_labels']}")
    print(f"Test labels ({len(test_labels)}): {consistency_results['test_labels']}")
    
    if consistency_results['all_labels_consistent']:
        print("✓ All val/test labels are present in train set")
    else:
        print("WARNING: Inconsistent labels found!")
        if not val_labels_in_train:
            print(f"  Val labels not in train: {val_labels - train_labels}")
        if not test_labels_in_train:
            print(f"  Test labels not in train: {test_labels - train_labels}")
    
    return consistency_results


def run_random_baseline(test_labels):
    """Run random baseline"""
    print("\nRunning random baseline...")
    
    # Generate random predictions
    unique_labels = np.unique(test_labels)
    random_preds = np.random.choice(unique_labels, size=len(test_labels))
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, random_preds)
    macro_f1 = f1_score(test_labels, random_preds, average='macro')
    balanced_acc = balanced_accuracy_score(test_labels, random_preds)
    
    random_results = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'balanced_accuracy': balanced_acc
    }
    
    print(f"Random baseline metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    
    return random_results


def run_majority_baseline(train_labels, test_labels):
    """Run majority class baseline"""
    print("\nRunning majority class baseline...")
    
    # Find majority class in train set
    majority_class = pd.Series(train_labels).mode()[0]
    
    # Generate majority class predictions
    majority_preds = np.full(len(test_labels), majority_class)
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, majority_preds)
    macro_f1 = f1_score(test_labels, majority_preds, average='macro')
    balanced_acc = balanced_accuracy_score(test_labels, majority_preds)
    
    majority_results = {
        'majority_class': majority_class,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'balanced_accuracy': balanced_acc
    }
    
    print(f"Majority baseline metrics:")
    print(f"  Majority class: {majority_class}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    
    return majority_results


def load_data(csv_path, language):
    """Load and split data by language"""
    full_df = pd.read_csv(csv_path)
    
    # Filter data by language
    if language == 'en':
        # For English, filter where language is NaN or 'en'
        lang_df = full_df[(full_df['language'].isna()) | (full_df['language'] == 'en')]
    else:
        # For other languages, filter by exact match
        lang_df = full_df[full_df['language'] == language]
    
    # Use consistent label column name
    label_col = 'Diagnosis Category' if 'Diagnosis Category' in lang_df.columns else 'label'
    
    # Create label mapping
    unique_labels = sorted(lang_df[label_col].unique())
    label2id = {str(label): i for i, label in enumerate(unique_labels)}
    id2label = {i: str(label) for label, i in label2id.items()}
    
    # Convert labels to ids
    lang_df['label_id'] = lang_df[label_col].astype(str).map(label2id)
    
    # Split data (80% train, 10% val, 10% test)
    from sklearn.model_selection import train_test_split
    
    # First split train vs temp
    train_df, temp_df = train_test_split(
        lang_df,
        test_size=0.2,
        random_state=42,
        stratify=lang_df['label_id']
    )
    
    # Then split temp into val and test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['label_id']
    )
    
    return train_df, val_df, test_df, label_col, label2id, id2label


def main():
    parser = argparse.ArgumentParser(description="Sanity check for baseline experiments")
    parser.add_argument('--csv_path', type=str, default='data/processed/multi_language_balanced_dataset.csv',
                        help="Path to CSV data file")
    parser.add_argument('--languages', type=str, nargs='+', default=['en', 'hi', 'pa'],
                        help="Languages to check")
    parser.add_argument('--output_dir', type=str, default='results',
                        help="Output directory for results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run sanity checks for each language
    all_sanity_results = {}
    
    for lang in args.languages:
        print(f"\n{'='*50}")
        print(f"Language: {lang.upper()}")
        print(f"{'='*50}")
        
        # Load data
        train_df, val_df, test_df, label_col, label2id, id2label = load_data(args.csv_path, lang)
        
        print(f"Data loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        print(f"Label column: {label_col}")
        print(f"Labels: {sorted(label2id.items())}")
        
        # Run sanity checks
        overlap_results = check_data_overlap(train_df, val_df, test_df)
        consistency_results = check_label_consistency(train_df, val_df, test_df, label_col)
        random_results = run_random_baseline(test_df['label_id'].tolist())
        majority_results = run_majority_baseline(train_df['label_id'].tolist(), test_df['label_id'].tolist())
        
        # Store results
        all_sanity_results[lang] = {
            'data_info': {
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'test_samples': len(test_df),
                'label_col': label_col,
                'num_labels': len(label2id),
                'label2id': label2id,
                'id2label': id2label
            },
            'overlap': overlap_results,
            'label_consistency': consistency_results,
            'random_baseline': random_results,
            'majority_baseline': majority_results
        }
    
    # Save sanity report
    sanity_report_file = os.path.join(args.output_dir, 'eval_sanity_report.md')
    with open(sanity_report_file, 'w', encoding='utf-8') as f:
        f.write("# Evaluation Pipeline Sanity Report\n\n")
        f.write("## Summary\n\n")
        f.write("This report verifies the evaluation pipeline and data splits for baseline experiments.\n\n")
        
        for lang in args.languages:
            results = all_sanity_results[lang]
            
            f.write(f"### {lang.upper()}\n\n")
            f.write(f"#### Data Info\n")
            f.write(f"- Train samples: {results['data_info']['train_samples']}\n")
            f.write(f"- Val samples: {results['data_info']['val_samples']}\n")
            f.write(f"- Test samples: {results['data_info']['test_samples']}\n")
            f.write(f"- Number of labels: {results['data_info']['num_labels']}\n\n")
            
            f.write(f"#### Label Mapping\n")
            for label, idx in results['data_info']['label2id'].items():
                f.write(f"  - {idx}: {label}\n")
            f.write(f"\n")
            
            f.write(f"#### Data Overlap\n")
            if results['overlap']['has_overlap']:
                f.write(f"⚠️  **WARNING**: Data overlap found!\n")
                f.write(f"   - Train-Val: {results['overlap']['train_val_overlap']} samples\n")
                f.write(f"   - Train-Test: {results['overlap']['train_test_overlap']} samples\n")
                f.write(f"   - Val-Test: {results['overlap']['val_test_overlap']} samples\n")
            else:
                f.write(f"✅ No data overlap between splits\n")
            f.write(f"\n")
            
            f.write(f"#### Label Consistency\n")
            if results['label_consistency']['all_labels_consistent']:
                f.write(f"✅ All labels consistent across splits\n")
            else:
                f.write(f"⚠️  **WARNING**: Inconsistent labels\n")
            f.write(f"\n")
            
            f.write(f"#### Baseline Results\n")
            f.write(f"| Baseline | Accuracy | Macro F1 | Balanced Accuracy |\n")
            f.write(f"|----------|----------|----------|-------------------|\n")
            f.write(f"| Random | {results['random_baseline']['accuracy']:.4f} | {results['random_baseline']['macro_f1']:.4f} | {results['random_baseline']['balanced_accuracy']:.4f} |\n")
            f.write(f"| Majority | {results['majority_baseline']['accuracy']:.4f} | {results['majority_baseline']['macro_f1']:.4f} | {results['majority_baseline']['balanced_accuracy']:.4f} |\n")
            f.write(f"\n")
    
    # Save detailed results as JSON for debugging
    sanity_json_file = os.path.join(args.output_dir, 'sanity_check_results.json')
    with open(sanity_json_file, 'w') as f:
        json.dump(all_sanity_results, f, indent=2, default=str)
    
    print(f"\nSanity check completed!")
    print(f"Report saved to: {sanity_report_file}")
    print(f"Detailed results saved to: {sanity_json_file}")


if __name__ == "__main__":
    main()
