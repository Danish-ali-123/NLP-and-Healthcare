#!/usr/bin/env python3
"""
Test script to verify English label filtering works correctly.
"""

import pandas as pd
import re

# Load the CSV file
csv_path = 'data/processed/multi_language_balanced_dataset.csv'
df = pd.read_csv(csv_path)

print("="*80)
print("TESTING ENGLISH LABEL FILTERING")
print("="*80)

# Step 1: Filter for English data (NaN in language column)
en_df = df[df['language'].isna()]
print(f"Step 1: Filtered English data samples: {len(en_df)}")

# Step 2: Get all unique labels from English data
unique_labels = sorted(en_df['Diagnosis Category'].unique())
print(f"Step 2: All labels from English data: {len(unique_labels)}")
print(f"Labels: {unique_labels}")

# Step 3: Apply English label filtering pattern
en_pattern = r'^[A-Za-z\s-]+$'
en_labels = [label for label in unique_labels if re.match(en_pattern, label)]
print(f"\nStep 3: Filtered English labels: {len(en_labels)}")
print(f"English-only labels: {en_labels}")

# Step 4: Check for non-English labels in English data
non_en_labels = [label for label in unique_labels if not re.match(en_pattern, label)]
print(f"\nStep 4: Non-English labels in English data: {len(non_en_labels)}")
print(f"Non-English labels: {non_en_labels}")

# Step 5: Test the label2id mapping
print(f"\nStep 5: Creating label mapping")
label2id = {label: i for i, label in enumerate(en_labels)}
id2label = {str(i): label for label, i in label2id.items()}
print(f"Label2ID mapping: {label2id}")
print(f"ID2Label mapping: {id2label}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
