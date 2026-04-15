#!/usr/bin/env python3
"""
Test script to verify label filtering by language.
"""

import pandas as pd
import re

# Load the CSV file
csv_path = 'data/processed/multi_language_balanced_dataset.csv'
df = pd.read_csv(csv_path)

# Filter for English language
en_df = df[df['language'] == 'en']
print(f"English data samples: {len(en_df)}")

# Get all unique labels from English data
unique_labels = sorted(en_df['Diagnosis Category'].unique())
print(f"All labels from English data: {len(unique_labels)}")
print(f"Labels: {unique_labels}")

# Test English label filtering pattern
en_pattern = r'^[A-Za-z\s-]+$'
en_labels = [label for label in unique_labels if re.match(en_pattern, label)]
print(f"\nFiltered English labels: {len(en_labels)}")
print(f"English-only labels: {en_labels}")

# Test Hindi label filtering pattern
hi_pattern = r'^[\u0900-\u097F\s-]+$'
hi_labels = [label for label in unique_labels if re.match(hi_pattern, label)]
print(f"\nHindi labels in English data: {len(hi_labels)}")
print(f"Hindi labels: {hi_labels}")

# Test Punjabi label filtering pattern
pa_pattern = r'^[\u0A00-\u0A7F\s-]+$'
pa_labels = [label for label in unique_labels if re.match(pa_pattern, label)]
print(f"\nPunjabi labels in English data: {len(pa_labels)}")
print(f"Punjabi labels: {pa_labels}")
