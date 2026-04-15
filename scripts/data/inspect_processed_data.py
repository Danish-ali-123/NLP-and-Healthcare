"""
Inspect processed data files to understand the dataset structure.

This script loads train.csv, val.csv, and test.csv and prints:
- Basic shapes and column information
- Language distribution
- Label distribution per language (top 20 labels)
"""

from pathlib import Path
import pandas as pd
import sys

# Get project root (2 levels up from scripts/data/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import safe_read_csv helper
from src.utils.io import safe_read_csv

processed_dir = PROJECT_ROOT / "data" / "processed"

# Load processed data splits with UTF-8 encoding
print("=" * 60)
print("LOADING PROCESSED DATA FILES")
print("=" * 60)
try:
    train, train_enc = safe_read_csv(processed_dir / "train.csv")
    val, val_enc = safe_read_csv(processed_dir / "val.csv")
    test, test_enc = safe_read_csv(processed_dir / "test.csv")
    print(f"✓ Loaded train.csv (encoding: {train_enc})")
    print(f"✓ Loaded val.csv (encoding: {val_enc})")
    print(f"✓ Loaded test.csv (encoding: {test_enc})")
except Exception as e:
    print(f"❌ Error loading CSV files: {e}")
    print(f"   Make sure to run: python -m src.data.preprocess")
    sys.exit(1)

# 1) Print basic shapes
print("\n" + "=" * 60)
print("BASIC SHAPES")
print("=" * 60)
print(f"Train shape: {train.shape}")
print(f"Val shape:   {val.shape}")
print(f"Test shape:  {test.shape}")

# 2) Check columns
print("\n" + "=" * 60)
print("COLUMNS")
print("=" * 60)
print("Columns:", list(train.columns))

# 3) Show language distribution
print("\n" + "=" * 60)
print("LANGUAGE DISTRIBUTION")
print("=" * 60)
print("\nTrain language distribution:")
print(train["language"].value_counts())
print("\nVal language distribution:")
print(val["language"].value_counts())
print("\nTest language distribution:")
print(test["language"].value_counts())

# 4) Show label distribution per language (top 20)
print("\n" + "=" * 60)
print("LABEL DISTRIBUTION PER LANGUAGE (TOP 20)")
print("=" * 60)

for split_name, df in [("train", train), ("val", val), ("test", test)]:
    print(f"\n=== {split_name.upper()} per-language label counts (top 20) ===")
    for lang in sorted(df["language"].unique()):
        sub = df[df["language"] == lang]
        print(f"\n[{split_name}] language = {lang}")
        print(sub["label"].value_counts().head(20))

# 5) Show unique label counts per language
print("\n" + "=" * 60)
print("UNIQUE LABEL COUNTS PER LANGUAGE")
print("=" * 60)
for split_name, df in [("train", train), ("val", val), ("test", test)]:
    print(f"\n{split_name.upper()}:")
    for lang in sorted(df["language"].unique()):
        sub = df[df["language"] == lang]
        unique_labels = sub["label"].nunique()
        print(f"  {lang}: {unique_labels} unique labels")

# 6) Show total samples per language
print("\n" + "=" * 60)
print("TOTAL SAMPLES PER LANGUAGE")
print("=" * 60)
for split_name, df in [("train", train), ("val", val), ("test", test)]:
    print(f"\n{split_name.upper()}:")
    lang_counts = df["language"].value_counts()
    for lang, count in lang_counts.items():
        print(f"  {lang}: {count} samples")

# 7) Check for malformed labels and Unknown labels
print("\n" + "=" * 60)
print("LABEL CLEANUP VERIFICATION")
print("=" * 60)

# Check for malformed Punjabi label
malformed_label = "ਮਸੂਕਲੋਸਕੇਲਟORG_REPLACEMENTਿਕਾਰ"
for split_name, df in [("train", train), ("val", val), ("test", test)]:
    pa_df = df[df["language"] == "pa"]
    if malformed_label in pa_df["label"].values:
        count = (pa_df["label"] == malformed_label).sum()
        print(f"⚠️  WARNING: Found {count} instances of malformed label '{malformed_label}' in {split_name}")
    else:
        print(f"✓ No malformed Punjabi label found in {split_name}")

# Check for Unknown labels (should be merged into Other)
unknown_labels = {
    "en": "Unknown",
    "hi": None,  # Hindi doesn't have Unknown
    "pa": "ਅਗਿਆਤ"
}

for split_name, df in [("train", train), ("val", val), ("test", test)]:
    for lang, unknown_label in unknown_labels.items():
        if unknown_label is None:
            continue
        lang_df = df[df["language"] == lang]
        if unknown_label in lang_df["label"].values:
            count = (lang_df["label"] == unknown_label).sum()
            print(f"⚠️  WARNING: Found {count} instances of 'Unknown' label '{unknown_label}' in {split_name} ({lang})")
        else:
            print(f"✓ No 'Unknown' label found in {split_name} ({lang})")

# 8) Verify label consistency across languages
print("\n" + "=" * 60)
print("LABEL CONSISTENCY CHECK")
print("=" * 60)

# Get unique labels per language
for split_name, df in [("train", train), ("val", val), ("test", test)]:
    print(f"\n{split_name.upper()}:")
    lang_labels = {}
    for lang in sorted(df["language"].unique()):
        lang_df = df[df["language"] == lang]
        unique_labels = sorted(lang_df["label"].unique())
        lang_labels[lang] = unique_labels
        print(f"  {lang}: {len(unique_labels)} unique labels")
    
    # Check if all languages have the same number of labels
    label_counts = [len(labels) for labels in lang_labels.values()]
    if len(set(label_counts)) == 1:
        print(f"  ✓ All languages have the same number of labels ({label_counts[0]})")
    else:
        print(f"  ⚠️  WARNING: Languages have different numbers of labels: {dict(zip(lang_labels.keys(), label_counts))}")

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)

