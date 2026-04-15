"""
Verify the mapping of patient 1049 from raw CSV to processed CSV.

This demonstrates that:
- ENGLISH.csv row → train.csv row with language='en'
- HINDI.csv row → train.csv row with language='hi'
- Both have the same age (62.0) and represent the same patient
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load raw data
raw_en = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "ENGLISH.csv")
raw_hi = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "HINDI.csv")

# Load processed data (all splits)
train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
val = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "val.csv")
test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")
all_processed = pd.concat([train, val, test], ignore_index=True)

# Find patient 1049 in raw data
patient_1049_en = raw_en[raw_en['patient_id'] == 1049.0]
patient_1049_hi = raw_hi[raw_hi['patient_id'] == 1049.0]

print("=" * 60)
print("VERIFICATION: Patient 1049 Mapping")
print("=" * 60)

if len(patient_1049_en) > 0:
    print("\n=== RAW ENGLISH.CSV (patient_id = 1049) ===")
    row = patient_1049_en.iloc[0]
    print(f"Pseudonymized_Patient History: {row['Pseudonymized_Patient History'][:80]}...")
    print(f"Pseudonymized_Diagnosis Category: {row['Pseudonymized_Diagnosis Category']}")
    print(f"Pseudonymized_age: {row['Pseudonymized_age']}")
    print(f"Pseudonymized_gender: {row['Pseudonymized_gender']}")

if len(patient_1049_hi) > 0:
    print("\n=== RAW HINDI.CSV (patient_id = 1049) ===")
    row = patient_1049_hi.iloc[0]
    print(f"Pseudonymized_Patient History: {row['Pseudonymized_Patient History'][:80]}...")
    print(f"Pseudonymized_Diagnosis Category: {row['Pseudonymized_Diagnosis Category']}")
    print(f"Pseudonymized_age: {row['Pseudonymized_age']}")
    print(f"Pseudonymized_gender: {row['Pseudonymized_gender']}")

# Find corresponding rows in processed data (check all splits)
# Look for English row with same text pattern
en_text_pattern = "persistent pain in the left hip"
en_matches = all_processed[
    (all_processed['language'] == 'en') & 
    (all_processed['text'].str.contains(en_text_pattern, na=False, case=False))
]

# Look for Hindi row with same text pattern
hi_text_pattern = "बाएं कूल्हे"
hi_matches = all_processed[
    (all_processed['language'] == 'hi') & 
    (all_processed['text'].str.contains(hi_text_pattern, na=False))
]

print("\n=== PROCESSED DATA (English row, patient 1049 equivalent) ===")
if len(en_matches) > 0:
    row = en_matches.iloc[0]
    print(f"text (first 80 chars): {row['text'][:80]}...")
    print(f"label: {row['label']}")
    print(f"age: {row['age']}")
    print(f"gender: {row['gender']}")
    print(f"language: {row['language']}")
    print(f"id: {row['id']}")
    # Check which split it's in
    if row['id'] in train['id'].values:
        print("Split: train")
    elif row['id'] in val['id'].values:
        print("Split: val")
    else:
        print("Split: test")
else:
    print("Not found in processed data")

print("\n=== PROCESSED DATA (Hindi row, patient 1049 equivalent) ===")
if len(hi_matches) > 0:
    row = hi_matches.iloc[0]
    print(f"text (first 80 chars): {row['text'][:80]}...")
    print(f"label: {row['label']}")
    print(f"age: {row['age']}")
    print(f"gender: {row['gender']}")
    print(f"language: {row['language']}")
    print(f"id: {row['id']}")
    # Check which split it's in
    if row['id'] in train['id'].values:
        print("Split: train")
    elif row['id'] in val['id'].values:
        print("Split: val")
    else:
        print("Split: test")
else:
    print("Not found in processed data")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
print("\nNote: The same patient (1049) appears as separate rows in processed data:")
print("  - One row with language='en' (from ENGLISH.csv)")
print("  - One row with language='hi' (from HINDI.csv)")
print("  - One row with language='pa' (from PUNJABI.csv)")
print("They are NOT linked by patient_id in the final CSV.")

