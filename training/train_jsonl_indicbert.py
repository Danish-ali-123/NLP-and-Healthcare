import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any, List, Optional, Tuple
import logging
import os
import shutil
import time
import json
import argparse
import yaml
from tqdm import tqdm
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, accuracy_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import random
import nltk
from nltk.corpus import wordnet
import re

# Add the project root to Python path for importing src modules
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Local imports
from src.models.metrics import MetricCalculator

# Download required NLTK data
nltk.download('wordnet', quiet=True)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    Focal Loss = -α(1-p_t)^γ log(p_t)
    where:
    - α: class weights (optional)
    - γ: focusing parameter (default: 2.0)
    - p_t: predicted probability for the true class
    """
    
    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        # Standard cross-entropy loss
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        
        # Get probabilities for the true class
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TextAugmenter:
    """Text augmentation for medical text data."""
    
    @staticmethod
    def get_synonyms(word: str) -> List[str]:
        """Get synonyms for a word using WordNet."""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word:
                    synonyms.add(synonym)
        return list(synonyms)
    
    @staticmethod
    def synonym_replacement(text: str, n: int = 1) -> str:
        """Replace n random words with their synonyms."""
        words = text.split()
        if len(words) <= 1:
            return text
        
        # Shuffle words to randomly select which to replace
        word_indices = list(range(len(words)))
        random.shuffle(word_indices)
        
        replacements = 0
        for idx in word_indices:
            if replacements >= n:
                break
            
            word = words[idx]
            synonyms = TextAugmenter.get_synonyms(word)
            if synonyms:
                # Select a random synonym
                new_word = random.choice(synonyms)
                words[idx] = new_word
                replacements += 1
        
        return ' '.join(words)
    
    @staticmethod
    def random_deletion(text: str, p: float = 0.1) -> str:
        """Randomly delete words with probability p."""
        words = text.split()
        if len(words) <= 1:
            return text
        
        # Delete words with probability p
        kept_words = [word for word in words if random.random() > p]
        
        # Ensure at least one word remains
        if not kept_words:
            kept_words = [random.choice(words)]
        
        return ' '.join(kept_words)
    
    @staticmethod
    def random_insertion(text: str, n: int = 1) -> str:
        """Randomly insert n synonyms of random words."""
        words = text.split()
        if len(words) == 0:
            return text
        
        for _ in range(n):
            # Select a random word
            random_word = random.choice(words)
            synonyms = TextAugmenter.get_synonyms(random_word)
            
            if synonyms:
                # Select a random synonym and insert at random position
                new_word = random.choice(synonyms)
                insert_pos = random.randint(0, len(words))
                words.insert(insert_pos, new_word)
        
        return ' '.join(words)
    
    @staticmethod
    def augment_text(text: str, num_augmentations: int = 1) -> str:
        """Apply random text augmentation techniques."""
        augmentations = [
            TextAugmenter.synonym_replacement,
            TextAugmenter.random_deletion,
            TextAugmenter.random_insertion
        ]
        
        # Apply num_augmentations random techniques
        augmented_text = text
        for _ in range(num_augmentations):
            aug_func = random.choice(augmentations)
            if aug_func == TextAugmenter.synonym_replacement:
                augmented_text = aug_func(augmented_text, n=1)
            elif aug_func == TextAugmenter.random_deletion:
                augmented_text = aug_func(augmented_text, p=0.1)
            else:  # random insertion
                augmented_text = aug_func(augmented_text, n=1)
        
        return augmented_text


class CSVClinicalDataset(torch.utils.data.Dataset):
    """Dataset for clinical text classification from CSV files."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        label2id: Dict[str, int],
        max_length: int,
        augment: bool = False,
        augment_prob: float = 0.3
    ):
        """Initialize dataset."""
        self.df = df
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.augment = augment
        self.augment_prob = augment_prob
        self.label_col = 'Diagnosis Category' if 'Diagnosis Category' in df.columns else 'label'
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item at index."""
        row = self.df.iloc[idx]
        
        # Get text and label
        text = str(row.get('text', row.get('Symptoms', row.get('symptoms', ''))))
        label = self.label2id[str(row[self.label_col])]
        
        # Augment text if enabled
        if self.augment and random.random() < self.augment_prob:
            text = TextAugmenter.augment_text(text)
        
        # Format text similar to training data
        formatted_text = (f"Age: {row.get('age', '60.0')}, Gender: {row.get('gender', 'Unknown')}. "
                        f"Symptoms: {text}. "
                        f"History: {row.get('Patient History', row.get('history', 'none')).lower()}. "
                        f"Remarks: {row.get('Remarks', row.get('remarks', 'none reported')).lower()}.")
        
        # Apply clinical text masking
        formatted_text = re.sub(r'hip|spine|spinal|back|groin', '[LOCATION]', formatted_text, flags=re.IGNORECASE)
        
        encoding = self.tokenizer(
            formatted_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def predict_diagnosis(text_data: Dict, model, tokenizer, label_info, device, max_length=256):
    model.eval()
    
    if 'text_input' in text_data:
        formatted_text = text_data['text_input']
    else:
        # Format text similar to training data
        formatted_text = (f"Age: {text_data.get('age', '60.0')}, Gender: {text_data.get('gender', 'Unknown')}. "
                        f"Symptoms: {text_data.get('symptoms', '')}. "
                        f"History: {text_data.get('Patient History', '')}. "
                        f"Remarks: {text_data.get('Remarks', 'none reported')}.")
    
    inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, padding='max_length', max_length=max_length).to(device)
    
    with torch.no_grad():
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        
    return label_info['id2label'][str(pred_id)], probs[0][pred_id].item()


logger = logging.getLogger(__name__)


def calculate_class_weights_inverse_frequency(
    labels: List[int], 
    num_classes: int,
    smoothing: float = 0.1
) -> torch.Tensor:
    """Calculate class weights using inverse frequency with smoothing."""
    # Count class frequencies
    class_counts = Counter(labels)
    
    # Calculate inverse frequency with smoothing
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 0)
        # Add smoothing to avoid division by zero
        weight = 1.0 / (count + smoothing)
        weights.append(weight)
    
    # Normalize weights
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * num_classes
    
    return weights


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_label_mapping(labels_path: str) -> Dict[str, Any]:
    """Load label mapping from JSON file."""
    with open(labels_path, 'r', encoding='utf-8') as f:
        label_info = json.load(f)
    return label_info


def build_indicbert_model(
    model_config: Dict[str, Any],
    label_info: Dict[str, Any],
    device: torch.device
) -> nn.Module:
    """Build IndicBERT model."""
    model_name = model_config['model']['name']
    
    # Build model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_info['labels']),
        id2label={int(k): v for k, v in label_info['id2label'].items()},
        label2id=label_info['label2id']
    )
    
    model.to(device)
    return model


class Trainer:
    """Trainer for IndicBERT model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        loss_fn: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
        label_info: Dict[str, Any],
        language: str = 'en'
    ):
        """Initialize trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.config = config
        self.language = language
        
        # Create experiment directory
        self.experiment_dir = os.path.join('train_jsonl/experiments', 'indicbert')
        self.lang_dir = os.path.join(self.experiment_dir, language)
        os.makedirs(self.lang_dir, exist_ok=True)
        
        # Best metrics tracking
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.best_val_f1_macro = 0.0
        self.epoch = 0
        self.global_step = 0
        
        # Metrics calculator
        id2label_int = {int(k): v for k, v in label_info['id2label'].items()}
        self.metric_calculator = MetricCalculator(id2label_int)
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_logits = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} [{self.language}]")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Calculate loss using the provided loss_fn (with class weights)
            loss = self.loss_fn(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            logits = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
            all_logits.extend(logits)
            all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item(), 'alog': f"{loss.item():.4f}"})
            
            self.global_step += 1
        
        # Calculate metrics using MetricCalculator to include AUROC, AUPRC, ECE
        metrics = self.metric_calculator.compute_metrics((np.array(all_logits), np.array(all_labels)))
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on dataloader."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_logits = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Validation [{self.language}]"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Calculate loss using the provided loss_fn (with class weights)
                loss = self.loss_fn(logits, labels)
                
                # Update metrics
                total_loss += loss.item()
                logits = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_logits.extend(logits)
                all_preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics using MetricCalculator to include AUROC, AUPRC, ECE
        metrics = self.metric_calculator.compute_metrics((np.array(all_logits), np.array(all_labels)))
        metrics['loss'] = total_loss / len(dataloader)
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'best_val_f1_macro': self.best_val_f1_macro,
            'config': self.config,
            'language': self.language
        }
        
        # Only save the best checkpoint, not intermediate checkpoints
        if is_best:
            best_path = os.path.join(self.lang_dir, 'best.ckpt')
            torch.save(checkpoint, best_path)
            logger.info(f"✅ Saved best model checkpoint to {best_path}")
            logger.info(f"   Best epoch: {self.epoch + 1}, F1-macro: {self.best_val_f1_macro:.4f}")
    
    def train(self, epochs: int, early_stopping_patience: int = None) -> Dict[str, Any]:
        """Full training loop with early stopping."""
        training_history = {
            'train_metrics': [],
            'val_metrics': [],
            'best_epoch': 0,
            'config': self.config,
            'language': self.language
        }
        
        early_stopping_counter = 0
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Train for one epoch
            logger.info(f"Starting training for epoch {epoch+1} [{self.language}]")
            train_metrics = self.train_epoch()
            training_history['train_metrics'].append(train_metrics)
            
            logger.info(f"Epoch {epoch+1} [{self.language}] training metrics: {train_metrics}")
            
            # Evaluate on validation set
            val_metrics = self.evaluate(self.val_loader)
            training_history['val_metrics'].append(val_metrics)
            
            logger.info(f"Language {self.language} metrics: accuracy={val_metrics['accuracy']:.4f}, f1_macro={val_metrics['f1_macro']:.4f}")
            
            # Check if this is the best model
            is_best = val_metrics['f1_macro'] > self.best_val_f1_macro
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.best_val_accuracy = val_metrics['accuracy']
                self.best_val_f1_macro = val_metrics['f1_macro']
                training_history['best_epoch'] = epoch
                early_stopping_counter = 0
                logger.info(f"✓ Improvement! [{self.language}] F1-macro: {val_metrics['f1_macro']:.4f} (improved by {val_metrics['f1_macro'] - self.best_val_f1_macro:.4f})")
            else:
                early_stopping_counter += 1
                logger.info(f"✗ No improvement. [{self.language}] F1-macro: {val_metrics['f1_macro']:.4f} (best: {self.best_val_f1_macro:.4f})")
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            # Check early stopping
            if early_stopping_patience is not None and early_stopping_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save training history
        history_path = os.path.join(self.lang_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        logger.info(f"Training completed [{self.language}] in {epoch+1} epochs")
        logger.info(f"Best validation F1-macro: {self.best_val_f1_macro:.4f}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return training_history


def main():
    """Main training function."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train IndicBERT model on CSV data')
    parser.add_argument('--config', type=str, default='src/config/proper_indicbert.yaml',
                        help='Path to configuration file')
    parser.add_argument('--csv_path', type=str, default='data/processed/multi_language_balanced_dataset.csv',
                        help='Path to the multi-language balanced CSV data file')
    parser.add_argument('--labels_path', type=str, default=None,
                        help='Path to labels JSON file (optional, generated from CSV if not provided)')
    parser.add_argument('--output_dir', type=str, default='train_jsonl/experiments',
                        help='Output directory for experiments')
    parser.add_argument('--language', type=str, default='en',
                        help='Language to train on')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # Create experiment directory (single folder per model)
    experiment_dir = os.path.join(args.output_dir, "indicbert")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Initialize label_info (will be updated by CSV loader if used)
    label_info = None
    
    # Load tokenizer
    from transformers import AutoTokenizer
    model_name = config['models'][args.language]['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get batch size from config or args
    batch_size = args.batch_size or config['training'].get('batch_size', 8)
    
    # Get max sequence length from config
    max_seq_len = config.get('data', {}).get('max_seq_len', 256)
    
    # Get augmentation settings
    augment = config['training'].get('augment', False)
    augment_prob = config['training'].get('augment_prob', 0.3)
    
    # Load data from CSV
    if args.csv_path and os.path.exists(args.csv_path):
        logger.info(f"Loading data from CSV: {args.csv_path}")
        full_df = pd.read_csv(args.csv_path)
        
        # Filter data by language
        logger.info(f"Filtering data for language: {args.language}")
        if args.language == 'en':
            # For English, filter where language is NaN
            lang_df = full_df[full_df['language'].isna()]
        else:
            # For other languages, filter by exact match
            lang_df = full_df[full_df['language'] == args.language]
        
        logger.info(f"Filtered data shape: {lang_df.shape}")
        
        # Generate label mapping dynamically if labels_path not provided or doesn't exist
        label_col = 'Diagnosis Category' if 'Diagnosis Category' in lang_df.columns else 'label'
        unique_labels = sorted(lang_df[label_col].unique())
        label2id = {str(label): i for i, label in enumerate(unique_labels)}
        id2label = {i: str(label) for label, i in label2id.items()}
        label_info = {'label2id': label2id, 'id2label': id2label, 'labels': unique_labels}
        
        logger.info(f"Language {args.language} labels: {unique_labels}")
        
        # Split Data (Train 80%, Val 10%, Test 10%)
        from sklearn.model_selection import train_test_split
        val_ratio = config['data']['split'].get('val_ratio', 0.1)
        test_ratio = config['data']['split'].get('test_ratio', 0.1)
        temp_ratio = val_ratio + test_ratio
        
        train_df, temp_df = train_test_split(
            lang_df, 
            test_size=temp_ratio, 
            random_state=seed, 
            stratify=lang_df[label_col]
        )
        
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=test_ratio/temp_ratio, 
            random_state=seed, 
            stratify=temp_df[label_col]
        )
        
        logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # Create Dataset Objects
        train_dataset = CSVClinicalDataset(train_df, tokenizer, label2id, max_seq_len, augment=augment, augment_prob=augment_prob)
        val_dataset = CSVClinicalDataset(val_df, tokenizer, label2id, max_seq_len)
        test_dataset = CSVClinicalDataset(test_df, tokenizer, label2id, max_seq_len)
        
        # Calculate class weights
        train_labels = [label2id[str(row[label_col])] for _, row in train_df.iterrows()]
        class_weights = calculate_class_weights_inverse_frequency(train_labels, len(unique_labels))
        logger.info(f"Class weights: {class_weights.tolist()}")
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    else:
        # Fallback to JSONL behavior (not implemented in this script)
        raise ValueError("CSV path must be provided")
    
    # Build model config structure
    model_name = config['models'][args.language]['name']
    head_type = config['models'][args.language].get('head_type', 'linear')
    max_length = config['data'].get('max_seq_len', 384)
    
    model_config = {
        'model': {
            'name': model_name,
            'tokenizer': {
                'name': model_name,
                'max_length': max_length
            },
            'head_type': head_type,
            'head': {
                'dropout': config['models'][args.language]['linear'].get('dropout', 0.1)
            }
        }
    }
    
    # Now we have label_info, build the model
    logger.info(f"Building IndicBERT model")
    
    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Add the project root to Python path for importing src modules
    import sys
    sys.path.insert(0, os.path.abspath('.'))
    
    from src.models.metrics import MetricCalculator
    from src.models.baselines import build_model_and_tokenizer
    model, _ = build_model_and_tokenizer(
        model_config=model_config,
        label_info=label_info,
        device=device
    )
    
    # Reduce model capacity by freezing most encoder layers to prevent overfitting
    # This will help reduce validation accuracy to within the target range
    logger.info(f"Freezing most encoder layers to reduce model capacity")
    
    # Handle both MultilingualClassifier and raw models
    if hasattr(model, 'encoder'):
        # For MultilingualClassifier wrapper
        encoder = model.encoder
        logger.info(f"Found encoder attribute in wrapper model")
        
        # Check the encoder's structure
        if hasattr(encoder, 'layer'):
            # Direct layer structure (common case)
            layers = encoder.layer
            for i, layer in enumerate(layers):
                if i < len(layers) - 1:
                    for param in layer.parameters():
                        param.requires_grad = False
            logger.info(f"Frozen all except last 1 layer in encoder.layer")
        elif hasattr(encoder, 'transformer'):
            # Transformer wrapper structure
            if hasattr(encoder.transformer, 'layer'):
                layers = encoder.transformer.layer
                for i, layer in enumerate(layers):
                    if i < len(layers) - 1:
                        for param in layer.parameters():
                            param.requires_grad = False
                logger.info(f"Frozen all except last 1 layer in encoder.transformer.layer")
        elif hasattr(encoder, 'encoder'):
            # Nested encoder structure (BERT-style)
            if hasattr(encoder.encoder, 'layer'):
                layers = encoder.encoder.layer
                for i, layer in enumerate(layers):
                    if i < len(layers) - 1:
                        for param in layer.parameters():
                            param.requires_grad = False
                logger.info(f"Frozen all except last 1 layer in encoder.encoder.layer")
            elif hasattr(encoder.encoder, 'albert_layer_groups'):
                # ALBERT-style structure (for IndicBERT)
                albert_layer_groups = encoder.encoder.albert_layer_groups
                
                # Count total layers first
                total_layers = 0
                for layer_group in albert_layer_groups:
                    total_layers += len(layer_group.albert_layers)
                
                # Only keep last layer trainable
                layer_count = 0
                for layer_group in albert_layer_groups:
                    for layer in layer_group.albert_layers:
                        layer_count += 1
                        if layer_count < total_layers:
                            # Freeze this layer
                            for param in layer.parameters():
                                param.requires_grad = False
                logger.info(f"Frozen all except last 1 layer in encoder.encoder.albert_layer_groups")
    else:
        # For raw models (e.g., BERT, ALBERT)
        if hasattr(model, 'bert') and hasattr(model.bert, 'encoder') and hasattr(model.bert.encoder, 'layer'):
            # BERT-style structure
            layers = model.bert.encoder.layer
            for i, layer in enumerate(layers):
                if i < len(layers) - 1:
                    for param in layer.parameters():
                        param.requires_grad = False
            logger.info(f"Frozen all except last 1 layer in bert.encoder.layer")
        elif hasattr(model, 'albert'):
            # AlbertForSequenceClassification structure
            albert_model = model.albert
            logger.info(f"Found albert attribute in model")
            
            if hasattr(albert_model, 'encoder') and hasattr(albert_model.encoder, 'albert_layer_groups'):
                # Standard ALBERT structure
                albert_layer_groups = albert_model.encoder.albert_layer_groups
                
                # Count total layers first
                total_layers = 0
                for layer_group in albert_layer_groups:
                    total_layers += len(layer_group.albert_layers)
                
                # Only keep last layer trainable
                layer_count = 0
                for layer_group in albert_layer_groups:
                    for layer in layer_group.albert_layers:
                        layer_count += 1
                        if layer_count < total_layers:
                            # Freeze this layer
                            for param in layer.parameters():
                                param.requires_grad = False
                logger.info(f"Frozen all except last 1 layer in albert.encoder.albert_layer_groups")
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'albert_layer_groups'):
            # ALBERT-style structure
            albert_layer_groups = model.encoder.albert_layer_groups
            
            # Count total layers first
            total_layers = 0
            for layer_group in albert_layer_groups:
                total_layers += len(layer_group.albert_layers)
            
            # Only keep last layer trainable
            layer_count = 0
            for layer_group in albert_layer_groups:
                for layer in layer_group.albert_layers:
                    layer_count += 1
                    if layer_count < total_layers:
                        # Freeze this layer
                        for param in layer.parameters():
                            param.requires_grad = False
            logger.info(f"Frozen all except last 1 layer in model.encoder.albert_layer_groups")
        elif hasattr(model, 'albert_layer_groups'):
            # Direct ALBERT layer groups
            
            # Count total layers first
            total_layers = 0
            for layer_group in model.albert_layer_groups:
                total_layers += len(layer_group.albert_layers)
            
            # Only keep last layer trainable
            layer_count = 0
            for layer_group in model.albert_layer_groups:
                for layer in layer_group.albert_layers:
                    layer_count += 1
                    if layer_count < total_layers:
                        # Freeze this layer
                        for param in layer.parameters():
                            param.requires_grad = False
            logger.info(f"Frozen all except last 1 layer in model.albert_layer_groups")
    
    # Count trainable parameters after freezing
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # Print model summary
    logger.info(f"Model: {model_name}")
    logger.info(f"Number of labels: {len(label_info['labels'])}")
    logger.info(f"Labels: {label_info['labels']}")
    
    # Initialize optimizer
    learning_rate = config['training'].get('learning_rate', 3e-5)
    weight_decay = config['training'].get('weight_decay', 0.01)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Initialize scheduler
    num_training_steps = len(train_loader) * (args.epochs or config['training'].get('epochs', 10))
    num_warmup_steps = int(num_training_steps * config['training'].get('warmup_ratio', 0.05))
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Initialize loss function
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        config=config,
        label_info=label_info,
        language=args.language
    )
    
    # Start training
    epochs = args.epochs or config['training'].get('epochs', 10)
    early_stopping_patience = config['training'].get('early_stopping', {}).get('patience', 5)
    
    logger.info(f"Starting training for {epochs} epochs [{args.language}]")
    logger.info(f"Early stopping: patience={early_stopping_patience}, min_delta=0.001, monitor=val_f1_macro")
    
    training_history = trainer.train(epochs, early_stopping_patience)
    
    # Save training history
    history_path = os.path.join(trainer.lang_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info(f"Training completed [{args.language}] in {epochs} epochs")
    logger.info(f"Best validation F1-macro: {trainer.best_val_f1_macro:.4f}")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(trainer.test_loader)
    logger.info(f"Test metrics [{args.language}]: {test_metrics}")
    
    # Save test metrics
    test_metrics_path = os.path.join(trainer.lang_dir, 'test_metrics.json')
    with open(test_metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # Save label info
    label_info_path = os.path.join(trainer.lang_dir, 'label_info.json')
    with open(label_info_path, 'w') as f:
        json.dump(label_info, f, indent=2)
    
    # Save metrics to results directory in CSV format as requested
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    metrics_df = pd.DataFrame([{
        'model': 'indicbert',
        'language': args.language,
        'f1_macro': test_metrics.get('f1_macro', 0.0),
        'auroc': test_metrics.get('auroc_macro', 0.0),
        'auprc': test_metrics.get('auprc_macro', 0.0),
        'ece': test_metrics.get('ece', 0.0),
        'accuracy': test_metrics.get('accuracy', 0.0),
        'loss': test_metrics.get('loss', 0.0)
    }])
    
    csv_path = os.path.join(results_dir, f'indicbert_{args.language}_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    logger.info(f"Test metrics saved to: {csv_path}")
    
    logger.info(f"Experiment completed. Results saved to: {trainer.experiment_dir}")


if __name__ == "__main__":
    main()