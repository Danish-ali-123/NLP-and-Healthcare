import torch
from torch.utils.data import Dataset
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Union
import json
import os
import random

logger = logging.getLogger(__name__)

class JSONLClinicalDataset(Dataset):
    """PyTorch Dataset for clinical data stored in JSONL format."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        label2id: Dict[str, int],
        max_length: int = 256,
        language: Optional[str] = None,
        augment: bool = False,
        augment_prob: float = 0.3
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.language = language
        self.augment = augment
        self.augment_prob = augment_prob
        
        # Load data
        self.data = self.load_data()
        
        logger.info(f"Loaded JSONL dataset with {len(self.data)} records from {data_path}")
        if language:
            logger.info(f"Filtered for language: {language}")
        if augment:
            logger.info(f"Text augmentation enabled with probability: {augment_prob}")
    
    def get_synonyms(self, word: str) -> List[str]:
        """Simple synonym getter using common medical terms."""
        medical_synonyms = {
            "pain": ["ache", "discomfort", "soreness", "tenderness"],
            "chronic": ["long-term", "persistent", "ongoing", "recurrent"],
            "acute": ["sudden", "severe", "sharp", "intense"],
            "patient": ["individual", "case", "client"],
            "history": ["background", "record", "medical history"],
            "female": ["woman", "lady", "female patient"],
            "male": ["man", "gentleman", "male patient"],
            "year": ["yr"],
            "years": ["yrs"],
            "old": ["age"],
            "hip": ["pelvis", "coxa"],
            "bone": ["osseous tissue", "skeletal structure"],
            "spinal": ["vertebral", "back"],
            "musculoskeletal": ["muscular-skeletal", "MSK"],
            "affected": ["involved", "impacted"],
            "area": ["region", "location", "site"]
        }
        return medical_synonyms.get(word.lower(), [])
    
    def augment_text(self, text: str) -> str:
        """Simple text augmentation for medical text."""
        words = text.split()
        if len(words) <= 1:
            return text
        
        # Randomly select augmentation type
        aug_type = random.choice(["synonym", "insert", "delete"])
        
        if aug_type == "synonym":
            # Replace one random word with synonym
            idx = random.randint(0, len(words) - 1)
            word = words[idx]
            synonyms = self.get_synonyms(word)
            if synonyms:
                words[idx] = random.choice(synonyms)
        
        elif aug_type == "insert":
            # Insert a random medical term
            medical_terms = ["medical", "clinical", "symptomatic", "diagnosed", "treated"]
            idx = random.randint(0, len(words))
            words.insert(idx, random.choice(medical_terms))
        
        elif aug_type == "delete":
            # Delete a random word with low probability
            if len(words) > 3:  # Don't delete if too short
                idx = random.randint(0, len(words) - 1)
                del words[idx]
        
        return " ".join(words)
    
    def load_data(self) -> pd.DataFrame:
        """Load data from JSONL file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        # Load JSONL file
        records = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        
        df = pd.DataFrame(records)
        
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
        
        # Apply text augmentation if enabled and random probability is met
        if self.augment and random.random() < self.augment_prob:
            text = self.augment_text(text)
        
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

def create_jsonl_data_loaders(
    jsonl_path: str,
    tokenizer: Any,
    label2id: Dict[str, int],
    batch_size: int = 16,
    max_length: int = 256,
    test_split: float = 0.1,
    val_split: float = 0.1,
    num_workers: int = 4,
    language: Optional[str] = None,
    augment: bool = False,
    augment_prob: float = 0.3
) -> Dict[str, torch.utils.data.DataLoader]:
    """Create data loaders for train, validation, and test sets from a single JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL data file
        tokenizer: Tokenizer to use for tokenization
        label2id: Label to ID mapping
        batch_size: Batch size for data loaders
        max_length: Maximum sequence length for tokenization
        test_split: Fraction of data to use for test set
        val_split: Fraction of data to use for validation set
        num_workers: Number of workers for data loaders
        language: Language to filter data by (optional)
        augment: Whether to enable text augmentation for training set
        augment_prob: Probability of applying augmentation to a sample
    
    Returns:
        Dictionary with 'train', 'val', and 'test' data loaders
    """
    
    from torch.utils.data import DataLoader, random_split
    
    # Create full dataset
    full_dataset = JSONLClinicalDataset(
        jsonl_path, tokenizer, label2id, max_length, language
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size
    
    # Split indices
    indices = list(range(total_size))
    random.shuffle(indices)
    
    # Create split indices
    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size+val_size]
    train_indices = indices[test_size+val_size:]
    
    # Create separate datasets for train, val, test
    # Training set with augmentation
    train_dataset = JSONLClinicalDataset(
        jsonl_path, tokenizer, label2id, max_length, language,
        augment=augment, augment_prob=augment_prob
    )
    train_dataset.data = train_dataset.data.iloc[train_indices].reset_index(drop=True)
    
    # Validation set without augmentation
    val_dataset = JSONLClinicalDataset(
        jsonl_path, tokenizer, label2id, max_length, language,
        augment=False
    )
    val_dataset.data = val_dataset.data.iloc[val_indices].reset_index(drop=True)
    
    # Test set without augmentation
    test_dataset = JSONLClinicalDataset(
        jsonl_path, tokenizer, label2id, max_length, language,
        augment=False
    )
    test_dataset.data = test_dataset.data.iloc[test_indices].reset_index(drop=True)
    
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
    
    logger.info(f"Created JSONL data loaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    logger.info(f"Split sizes: train={train_size}, val={val_size}, test={test_size}")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
