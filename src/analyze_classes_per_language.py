import pandas as pd

# Path to the dataset
dataset_path = "data/processed/multi_language_balanced_dataset.csv"

# Read the dataset
df = pd.read_csv(dataset_path)

# Convert NaN to 'en' for English records
df['language'] = df['language'].fillna('en')

# Map language codes to full names
language_names = {
    'en': 'English',
    'hi': 'Hindi', 
    'pa': 'Punjabi'
}

# Analyze classes per language
print("# Classes Per Language Analysis")
print()
print("| Language | Number of Classes | Class Names |")
print("|----------|-------------------|-------------|")

# Process each language
for lang_code, lang_name in language_names.items():
    # Filter records for this language
    lang_data = df[df['language'] == lang_code]
    
    # Get unique classes
    classes = lang_data['Diagnosis Category'].unique()
    num_classes = len(classes)
    
    # Sort classes alphabetically
    sorted_classes = sorted(classes)
    
    # Join class names with line breaks for table
    class_names = '\n'.join(sorted_classes)
    
    print(f"| {lang_name} | {num_classes} | {class_names} |")

print()
print("# Detailed Class Distribution")
print()

# For each language, show class counts
for lang_code, lang_name in language_names.items():
    print(f"## {lang_name} Classes:")
    lang_data = df[df['language'] == lang_code]
    class_counts = lang_data['Diagnosis Category'].value_counts().sort_index()
    
    for cls, count in class_counts.items():
        print(f"- {cls}: {count} records")
    print()

# Overall summary
print("# Overall Summary")
print(f"Total unique classes across all languages: {len(df['Diagnosis Category'].unique())}")
print(f"Total records: {len(df)}")
