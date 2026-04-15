import torch
from torch.utils.data import Dataset
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Union
import json
import os

logger = logging.getLogger(__name__)

class MultilingualClinicalDataset(Dataset):
    """PyTorch Dataset for multilingual clinical data."""
    
    def __init__(
        self, 
        data_path: str,
        tokenizer: Any,
        label2id: Dict[str, int],
        max_length: int = 256,
        language: Optional[str] = None
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.language = language
        
        # Load data
        self.data = self.load_data()
        
        logger.info(f"Loaded dataset with {len(self.data)} records from {data_path}")
        if language:
            logger.info(f"Filtered for language: {language}")
    
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        df = pd.read_csv(self.data_path)
        
        # Filter by language if specified
        if self.language:
            df = df[df['language'] == self.language]
            
        return df
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset."""
        row = self.data.iloc[idx]
        
        text = str(row['text'])
        label = str(row['label'])
        language = str(row['language'])
        
        # Tokenize text (no padding here - will be done dynamically in collator)
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None  # Return as lists, not tensors
        )
        
        # Convert label to ID
        label_id = self.label2id.get(label, -1)
        if label_id == -1:
            logger.warning(f"Label '{label}' not found in label2id mapping")
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': label_id,
            'language': language
        }

class DataCollator:
    """Data collator for multilingual clinical data with dynamic padding."""
    
    def __init__(self, tokenizer: Any, padding: str = 'longest'):
        """
        Args:
            tokenizer: Tokenizer to use for padding
            padding: Padding strategy ('longest' for dynamic padding, 'max_length' for fixed)
        """
        self.tokenizer = tokenizer
        self.padding = padding
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch of data with dynamic padding."""
        # Extract texts and labels
        texts = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        languages = [item['language'] for item in batch]
        
        # Pad sequences dynamically
        padded = self.tokenizer.pad(
            {'input_ids': texts, 'attention_mask': [[1] * len(ids) for ids in texts]},
            padding=self.padding,
            return_tensors='pt'
        )
        
        return {
            'input_ids': padded['input_ids'],
            'attention_mask': padded['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long),
            'language': languages
        }

def load_label_mapping(labels_path: str) -> Dict[str, Any]:
    """Load label mapping from JSON file."""
    with open(labels_path, 'r', encoding='utf-8') as f:
        label_info = json.load(f)
    return label_info

def create_data_loaders(
    train_path: str,
    val_path: str, 
    test_path: str,
    tokenizer: Any,
    label2id: Dict[str, int],
    batch_size: int = 16,
    max_length: int = 256,
    num_workers: int = 4
) -> Dict[str, torch.utils.data.DataLoader]:
    """Create data loaders for train, validation, and test sets."""
    
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = MultilingualClinicalDataset(
        train_path, tokenizer, label2id, max_length
    )
    val_dataset = MultilingualClinicalDataset(
        val_path, tokenizer, label2id, max_length  
    )
    test_dataset = MultilingualClinicalDataset(
        test_path, tokenizer, label2id, max_length
    )
    
    # Create data collator with dynamic padding
    data_collator = DataCollator(tokenizer, padding='longest')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=data_collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=data_collator
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers, 
        collate_fn=data_collator
    )
    
    logger.info(f"Created data loaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }