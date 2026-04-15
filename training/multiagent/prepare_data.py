import pandas as pd
import os
from pathlib import Path

"""
Script to prepare datasets for the multi-agent pipeline.
Transforms various dataset formats into the expected pipeline input format:
id, text, language, true_label
"""

def prepare_test_csv(input_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Prepare test.csv for pipeline input.
    This file already has the correct format, just needs column renaming.
    
    Args:
        input_path: Path to test.csv file
        output_path: Optional path to save prepared file
        
    Returns:
        Prepared DataFrame
    """
    df = pd.read_csv(input_path)
    
    # Rename columns to match pipeline expectations
    df = df.rename(columns={
        'label': 'true_label'
    })
    
    # Select only required columns
    df = df[['id', 'text', 'language', 'true_label']]
    
    if output_path:
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Saved prepared test data to: {output_path}")
    
    return df

def prepare_multi_language_dataset(input_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Prepare multi_language_balanced_dataset.csv for pipeline input.
    
    Args:
        input_path: Path to multi_language_balanced_dataset.csv file
        output_path: Optional path to save prepared file
        
    Returns:
        Prepared DataFrame
    """
    df = pd.read_csv(input_path)
    
    # Drop rows without language information
    df = df.dropna(subset=['language'])
    
    # Create id column
    df['id'] = df.index.map(lambda x: f"sample_{x}")
    
    # Rename columns
    df = df.rename(columns={
        'text_input': 'text',
        'Diagnosis Category': 'true_label'
    })
    
    # Select only required columns
    df = df[['id', 'text', 'language', 'true_label']]
    
    if output_path:
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Saved prepared multi-language data to: {output_path}")
    
    return df

def main():
    """Main function to prepare datasets."""
    # Define input paths
    test_path = "d:\\NLP_Healthcare_Project_Structure\\data\\processed\\test.csv"
    multi_lang_path = "d:\\NLP_Healthcare_Project_Structure\\data\\processed\\multi_language_balanced_dataset.csv"
    
    # Define output paths
    output_dir = "d:\\NLP_Healthcare_Project_Structure\\train_jsonl\\multiagent\\data"
    os.makedirs(output_dir, exist_ok=True)
    
    test_output = os.path.join(output_dir, "prepared_test.csv")
    multi_lang_output = os.path.join(output_dir, "prepared_multi_lang.csv")
    
    print("Preparing test.csv...")
    prepare_test_csv(test_path, test_output)
    
    print("Preparing multi_language_balanced_dataset.csv...")
    prepare_multi_language_dataset(multi_lang_path, multi_lang_output)
    
    print("Data preparation completed!")
    print(f"Prepared test data: {test_output}")
    print(f"Prepared multi-language data: {multi_lang_output}")

if __name__ == "__main__":
    main()
