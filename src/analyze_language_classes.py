import pandas as pd
import os

# Path to the CSV file
csv_path = 'data/processed/multi_language_balanced_dataset.csv'

# Read the CSV file
df = pd.read_csv(csv_path)

# Count classes for each language
print("=== Language-wise Class Count Analysis ===")
print()

# English: language column is empty (NaN)
english_df = df[df['language'].isna()]
english_classes = english_df['Diagnosis Category'].unique()
print(f"English:")
print(f"- Total samples: {len(english_df)}")
print(f"- Unique classes: {len(english_classes)}")
print(f"- Classes: {sorted(english_classes)}")
print()

# Hindi: language column is 'hi'
hindi_df = df[df['language'] == 'hi']
hindi_classes = hindi_df['Diagnosis Category'].unique()
print(f"Hindi:")
print(f"- Total samples: {len(hindi_df)}")
print(f"- Unique classes: {len(hindi_classes)}")
print(f"- Classes: {sorted(hindi_classes)}")
print()

# Punjabi: language column is 'pa'
punjabi_df = df[df['language'] == 'pa']
punjabi_classes = punjabi_df['Diagnosis Category'].unique()
print(f"Punjabi:")
print(f"- Total samples: {len(punjabi_df)}")
print(f"- Unique classes: {len(punjabi_classes)}")
print(f"- Classes: {sorted(punjabi_classes)}")
print()

# Overall analysis
print("=== Overall Analysis ===")
total_classes = df['Diagnosis Category'].unique()
print(f"Total unique classes in dataset: {len(total_classes)}")
print(f"Total samples: {len(df)}")
print()
print("Done!")