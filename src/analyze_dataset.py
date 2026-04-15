import pandas as pd

# Load the dataset
df = pd.read_csv('data/processed/multi_language_balanced_dataset.csv')

# Print basic information
print('Dataset shape:', df.shape)
print('Columns:', df.columns.tolist())

# Language distribution
print('\nLanguage distribution:')
print(df['language'].value_counts(dropna=False))

# Class distribution per language
print('\nClass distribution per language:')
for lang in ['en', 'hi', 'pa']:
    if lang == 'en':
        lang_df = df[df['language'].isna()]
    else:
        lang_df = df[df['language'] == lang]
    
    print(f'\n{lang}:')
    print(f'Total samples: {len(lang_df)}')
    print(lang_df['Diagnosis Category'].value_counts())
    print('\nPercentage distribution:')
    print((lang_df['Diagnosis Category'].value_counts(normalize=True) * 100).round(2))
