import pytest
import pandas as pd
import tempfile
import os
import json
from src.data.preprocess import DataPreprocessor
from src.data.utils import validate_processed_data

class TestPreprocessing:
    """Test data preprocessing functionality."""
    
    def test_sample_data_loading(self):
        """Test that sample data can be loaded."""
        # Test English sample
        en_path = "data/samples/sample_english.csv"
        if os.path.exists(en_path):
            df_en = pd.read_csv(en_path)
            assert not df_en.empty
            assert 'Pseudonymized_Patient History' in df_en.columns
            assert 'Pseudonymized_Diagnosis Category' in df_en.columns
            
        # Test Hindi sample  
        hi_path = "data/samples/sample_hindi.csv"
        if os.path.exists(hi_path):
            df_hi = pd.read_csv(hi_path)
            assert not df_hi.empty
            
        # Test Punjabi sample
        pa_path = "data/samples/sample_punjabi.csv"
        if os.path.exists(pa_path):
            df_pa = pd.read_csv(pa_path)
            assert not df_pa.empty
    
    def test_data_preprocessor_initialization(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor is not None
        assert 'ENGLISH' in preprocessor.languages
        assert preprocessor.text_column == "Pseudonymized_Patient History"
    
    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        preprocessor = DataPreprocessor()
        
        test_cases = [
            ("  Hello   World  ", "Hello World"),
            ("", ""),
            (None, ""),
            ("Normal text", "Normal text")
        ]
        
        for input_text, expected in test_cases:
            result = preprocessor.clean_text(input_text)
            assert result == expected

if __name__ == "__main__":
    pytest.main([__file__])