import pytest
import torch
import numpy as np
from src.models.baselines import MultilingualClassifier, build_model_and_tokenizer
from src.models.metrics import MetricCalculator

class TestModels:
    """Test model functionality."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = MultilingualClassifier(
            model_name='distilbert-base-uncased',
            num_labels=3,
            id2label={0: 'A', 1: 'B', 2: 'C'},
            label2id={'A': 0, 'B': 1, 'C': 2}
        )
        
        assert model is not None
        assert model.num_labels == 3
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = MultilingualClassifier(
            model_name='distilbert-base-uncased',
            num_labels=3,
            id2label={0: 'A', 1: 'B', 2: 'C'},
            label2id={'A': 0, 'B': 1, 'C': 2}
        )
        
        # Test input
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        
        outputs = model(input_ids, attention_mask)
        assert outputs.logits.shape == (2, 3)

class TestMetrics:
    """Test metrics functionality."""
    
    def test_metric_calculator(self):
        """Test metric calculator."""
        id2label = {0: 'A', 1: 'B', 2: 'C'}
        calculator = MetricCalculator(id2label)
        
        predictions = [0, 1, 2, 0, 1]
        labels = [0, 1, 2, 0, 1]
        languages = ['ENGLISH', 'ENGLISH', 'HINDI', 'HINDI', 'PUNJABI']
        
        metrics = calculator.compute_metrics((predictions, labels))
        assert 'accuracy' in metrics
        assert metrics['accuracy'] == 1.0
        
        lang_metrics = calculator.compute_per_language_metrics(predictions, labels, languages)
        assert 'ENGLISH' in lang_metrics
        assert 'HINDI' in lang_metrics
        assert 'PUNJABI' in lang_metrics

if __name__ == "__main__":
    pytest.main([__file__])