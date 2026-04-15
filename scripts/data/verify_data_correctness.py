"""
Verify data correctness:
- Check processed CSVs (text, labels, language distribution)
- Detect leakage (same patient_id across splits)
- Print label distributions and majority-class baselines
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from src.utils.io import safe_read_csv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def check_data_correctness():
    """Verify processed data correctness."""
    processed_dir = project_root / "data" / "processed"
    
    print("=" * 80)
    print("DATA CORRECTNESS VERIFICATION")
    print("=" * 80)
    
    # Load splits
    try:
        train_df, train_enc = safe_read_csv(processed_dir / "train.csv")
        val_df, val_enc = safe_read_csv(processed_dir / "val.csv")
        test_df, test_enc = safe_read_csv(processed_dir / "test.csv")
        print(f"✓ Loaded splits (encodings: train={train_enc}, val={val_enc}, test={test_enc})")
    except Exception as e:
        print(f"❌ Error loading CSV files: {e}")
        return False
    
    # A1: Check text column
    print("\n" + "=" * 80)
    print("A1: TEXT COLUMN VERIFICATION")
    print("=" * 80)
    
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n[{split_name}]")
        print(f"  Total rows: {len(df)}")
        print(f"  Non-empty text: {(df['text'].astype(str).str.strip() != '').sum()}")
        print(f"  Empty text: {(df['text'].astype(str).str.strip() == '').sum()}")
        text_lens = df['text'].astype(str).str.len()
        print(f"  Text length - mean: {text_lens.mean():.1f}, median: {text_lens.median():.1f}, min: {text_lens.min()}, max: {text_lens.max()}")
        
        if (df['text'].astype(str).str.strip() == '').sum() > 0:
            print(f"  ⚠️ WARNING: {split_name} has empty text rows!")
    
    # A2: Check label distribution per language
    print("\n" + "=" * 80)
    print("A2: LABEL DISTRIBUTION PER LANGUAGE PER SPLIT")
    print("=" * 80)
    
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n[{split_name}]")
        for lang in ['en', 'hi', 'pa']:
            lang_df = df[df['language'] == lang]
            if len(lang_df) == 0:
                print(f"  {lang}: NO DATA")
                continue
            
            unique_labels = sorted(lang_df['label'].unique())
            label_counts = lang_df['label'].value_counts().to_dict()
            
            print(f"  {lang}:")
            print(f"    Total samples: {len(lang_df)}")
            print(f"    Unique labels: {len(unique_labels)}")
            print(f"    Labels: {unique_labels}")
            print(f"    Label counts:")
            for label in unique_labels:
                count = label_counts.get(label, 0)
                pct = (count / len(lang_df)) * 100
                print(f"      {label}: {count} ({pct:.1f}%)")
            
            # Majority class baseline
            majority_label = max(label_counts.items(), key=lambda x: x[1])[0]
            majority_count = label_counts[majority_label]
            majority_ratio = majority_count / len(lang_df)
            
            print(f"    Majority class: {majority_label} ({majority_ratio:.1%})")
            
            # Calculate majority baseline metrics
            true_labels = lang_df['label'].values
            majority_predictions = np.array([majority_label] * len(true_labels))
            
            majority_acc = accuracy_score(true_labels, majority_predictions)
            majority_f1_macro = f1_score(true_labels, majority_predictions, average='macro', zero_division=0)
            
            print(f"    Majority baseline accuracy: {majority_acc:.4f}")
            print(f"    Majority baseline macro-F1: {majority_f1_macro:.4f}")
    
    # A3: Detect leakage (check if same ID appears in multiple splits)
    print("\n" + "=" * 80)
    print("A3: LEAKAGE DETECTION")
    print("=" * 80)
    
    # Check if 'id' column exists
    if 'id' in train_df.columns and 'id' in val_df.columns and 'id' in test_df.columns:
        train_ids = set(train_df['id'].astype(str))
        val_ids = set(val_df['id'].astype(str))
        test_ids = set(test_df['id'].astype(str))
        
        train_val_overlap = train_ids & val_ids
        train_test_overlap = train_ids & test_ids
        val_test_overlap = val_ids & test_ids
        
        print(f"Train IDs: {len(train_ids)}")
        print(f"Val IDs: {len(val_ids)}")
        print(f"Test IDs: {len(test_ids)}")
        print(f"Train-Val overlap: {len(train_val_overlap)} IDs")
        print(f"Train-Test overlap: {len(train_test_overlap)} IDs")
        print(f"Val-Test overlap: {len(val_test_overlap)} IDs")
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            print("⚠️ WARNING: LEAKAGE DETECTED! Same IDs appear in multiple splits.")
            if train_val_overlap:
                print(f"  Train-Val overlap examples: {list(train_val_overlap)[:5]}")
            if train_test_overlap:
                print(f"  Train-Test overlap examples: {list(train_test_overlap)[:5]}")
            if val_test_overlap:
                print(f"  Val-Test overlap examples: {list(val_test_overlap)[:5]}")
            print("\n⚠️ RECOMMENDATION: Use GROUP split by patient_id to prevent leakage.")
        else:
            print("✓ No leakage detected - IDs are unique across splits")
    else:
        print("⚠️ 'id' column not found - cannot check for leakage")
        print("  Recommendation: Add 'id' column during preprocessing")
    
    # A4: Language distribution
    print("\n" + "=" * 80)
    print("A4: LANGUAGE DISTRIBUTION")
    print("=" * 80)
    
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        lang_counts = df['language'].value_counts()
        print(f"\n[{split_name}]")
        for lang in ['en', 'hi', 'pa']:
            count = lang_counts.get(lang, 0)
            pct = (count / len(df)) * 100 if len(df) > 0 else 0
            print(f"  {lang}: {count} ({pct:.1f}%)")
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    success = check_data_correctness()
    sys.exit(0 if success else 1)

