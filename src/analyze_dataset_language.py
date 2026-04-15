import pandas as pd
import os

# Path to the dataset
dataset_path = "data/processed/multi_language_balanced_dataset.csv"

# Read the dataset
df = pd.read_csv(dataset_path)

# Count records per language
language_counts = df['language'].value_counts(dropna=False)

# Handle empty language values (nan = English)
language_counts = language_counts.rename(index={None: 'empty', 'nan': 'empty'})

# Convert NaN to 'en' for English records
df['language'] = df['language'].fillna('en')

# Recount with proper language codes
language_counts = df['language'].value_counts(dropna=False)

# Calculate total
total = len(df)

# Create a summary table
print("# Dataset Language Distribution")
print(f"Total records: {total}")
print()
print("| Language | Count | Percentage |")
print("|----------|-------|------------|")

for lang, count in language_counts.items():
    percentage = (count / total) * 100
    # Map language codes to full names
    if lang == 'en':
        lang_name = 'English'
    elif lang == 'hi':
        lang_name = 'Hindi'
    elif lang == 'pa':
        lang_name = 'Punjabi'
    elif lang == 'empty':
        lang_name = 'Empty'
    else:
        lang_name = str(lang)
    
    print(f"| {lang_name} | {count} | {percentage:.1f}% |")

print()
print("# Language Distribution Summary")
print(f"English: {language_counts.get('en', 0)} records")
print(f"Hindi: {language_counts.get('hi', 0)} records")
print(f"Punjabi: {language_counts.get('pa', 0)} records")
print(f"Empty: {language_counts.get('empty', 0)} records")
print(f"Total: {total} records")
