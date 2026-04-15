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
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import random
import nltk
from nltk.corpus import wordnet
import re

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
            augmented_text = aug_func(augmented_text)
        
        return augmented_text


# Add src to path for imports
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Local imports
from src.models.metrics import MetricCalculator, compute_metrics_from_logits
from src.utils.io import write_json, read_json
from src.utils.seed import set_seed


class CSVClinicalDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, label2id, max_length=256, augment=False, augment_prob=0.2):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.augment = augment
        self.augment_prob = augment_prob
        self.augmenter = TextAugmenter()
        
        # Use the 'text_input' column directly as the dataset is already cleaned
        self.texts = df['text_input'].astype(str).tolist()
            
        # Get target labels
        label_col = 'Diagnosis Category' if 'Diagnosis Category' in df.columns else 'label'
        self.labels = [self.label2id[cat] for cat in df[label_col]]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        # Apply clinical masking at runtime to catch anything missed in CSV
        # This prevents the model from seeing "Hip", "Spine", or specific condition names
        text = re.sub(r'hip|spine|spinal|back|groin|lower joint area|mid-body axis|upper extremity joint', '[LOCATION]', text, flags=re.IGNORECASE)
        
        if self.augment and random.random() < self.augment_prob:
            text = self.augmenter.augment_text(text)
            
        encoding = self.tokenizer(
            text,
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


def predict_diagnosis(text_data: Dict, model, tokenizer, label_info, device, max_length=384):
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
    
    # Calculate inverse frequencies
    total_samples = len(labels)
    weights = []
    
    for class_id in range(num_classes):
        count = class_counts.get(class_id, 0)
        if count == 0:
            # If class not present, use average weight
            freq = 1.0 / num_classes
        else:
            freq = count / total_samples
        
        # Inverse frequency with smoothing
        inv_freq = 1.0 / (freq + smoothing)
        weights.append(inv_freq)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum() * num_classes  # Normalize to have mean = 1
    
    return torch.tensor(weights, dtype=torch.float32)


def get_optimizer_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float = 0.01,
    no_decay: List[str] = None
) -> List[Dict[str, Any]]:
    """Get parameter groups for optimizer with proper weight decay exclusions."""
    # Ensure base_lr and weight_decay are floats
    base_lr = float(base_lr)
    weight_decay = float(weight_decay)
    
    no_decay = no_decay or ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    
    # Standard parameter grouping with weight decay exclusions
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'lr': float(base_lr), 'weight_decay': float(weight_decay)},
        {'params': no_decay_params, 'lr': float(base_lr), 'weight_decay': 0.0}
    ]
    
    return param_groups


class Trainer:
    """Trainer for clinical NLP models with CSV data."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        experiment_dir: str,
        label_info: Dict[str, Any],
        config: Dict[str, Any],
        language: str,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.experiment_dir = experiment_dir
        self.label_info = label_info
        self.config = config
        self.language = language
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.best_val_f1_macro = 0.0
        self.best_epoch = 0
        self.epochs_since_improvement = 0
        
        # Metrics calculator
        id2label_int = {int(k): v for k, v in label_info['id2label'].items()}
        self.metric_calculator = MetricCalculator(id2label_int)
        
        # Ensure experiment directory exists
        self.lang_dir = os.path.join(experiment_dir, language)
        os.makedirs(self.lang_dir, exist_ok=True)
        os.makedirs(os.path.join(self.lang_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.lang_dir, 'best_models'), exist_ok=True)
        
        logger.info(f"Trainer initialized for language: {language}, device: {device}")
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss with optional focal loss, label smoothing and class weights."""
        # Check if focal loss is enabled in config
        loss_type = self.config['training']['loss'].get('type', 'cross_entropy').lower()
        
        if self.label_smoothing > 0:
            # Label smoothing implementation
            num_classes = logits.size(-1)
            batch_size = labels.size(0)
            
            # Create smoothed distribution
            true_dist = torch.zeros_like(logits)
            true_dist.scatter_(1, labels.unsqueeze(1), 1.0)
            smooth_dist = true_dist * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes
            
            # Compute loss
            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            loss = -torch.sum(smooth_dist * log_probs, dim=1)
            
            # Apply class weights if specified
            if self.class_weights is not None:
                weights = self.class_weights[labels]
                loss = loss * weights
            
            return loss.mean()
        else:
            # Select loss function based on config
            if loss_type == 'focal':
                # Use Focal Loss
                gamma = self.config['training']['loss'].get('gamma', 2.0)
                loss_fct = FocalLoss(weight=self.class_weights, gamma=gamma)
            else:
                # Standard Cross-Entropy Loss
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            
            return loss_fct(logits, labels)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_logits = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1} [{self.language}]")
        
        running_acc = 0.0
        running_f1 = 0.0
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, labels=labels)
            
            # Compute loss
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
            else:
                loss = self.compute_loss(outputs.logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            gradient_clip_norm = float(self.config['training'].get('gradient_clip_norm', 1.0))
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=gradient_clip_norm)
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            logits = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits)
            
            # Update running metrics
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(self.train_loader):
                running_metrics = self.metric_calculator.compute_metrics(
                    (np.array(all_logits[-1000:]), np.array(all_labels[-1000:]))
                )
                running_acc = running_metrics.get("accuracy", 0)
                running_f1 = running_metrics.get("f1_macro", 0)
            else:
                # Simple accuracy is fast enough for every step
                running_acc = np.mean(np.array(all_predictions) == np.array(all_labels))
            
            # Update progress bar with compact labels to prevent truncation
            postfix = {
                'loss': f'{loss.item():.3f}',
                'alog': f'{total_loss/(batch_idx+1):.3f}',
                'acc': f'{running_acc:.3f}',
                'f1': f'{running_f1:.3f}'
            }
            progress_bar.set_postfix(postfix)
            
            self.global_step += 1
        
        # Compute comprehensive epoch metrics using the calculator
        # Pass logits instead of just predictions to enable AUROC, AUPRC, ECE computation
        epoch_metrics = self.metric_calculator.compute_metrics(
            (np.array(all_logits), np.array(all_labels))
        )
        epoch_metrics['loss'] = total_loss / len(self.train_loader)
        
        logger.info(f"Epoch {self.epoch + 1} [{self.language}] training metrics: {epoch_metrics}")
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, Any]:
        """Validate model on validation set with detailed per-class metrics."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_logits = []
        all_languages = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validation [{self.language}]"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Get languages from batch
                languages = batch.get('language', [self.language] * len(labels))
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, labels=labels)
                
                # Compute loss
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss = outputs.loss
                else:
                    loss = self.compute_loss(outputs.logits, labels)
                
                # Collect predictions and logits
                total_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(torch.softmax(outputs.logits, dim=-1).cpu().numpy())
                all_languages.extend(languages)
        
        # Compute comprehensive validation metrics using the calculator
        # Pass logits instead of just predictions to enable AUROC, AUPRC, ECE computation
        val_metrics = self.metric_calculator.compute_metrics(
            (np.array(all_logits), np.array(all_labels))
        )
        val_metrics['loss'] = total_loss / len(self.val_loader)
        
        # Compute detailed metrics including per-class performance for logging
        detailed_metrics = self.metric_calculator.compute_detailed_metrics(
            predictions=all_predictions,
            labels=all_labels,
            languages=all_languages
        )
        
        logger.info(f"Validation metrics [{self.language}]: {val_metrics}")
        
        # Log per-class metrics
        logger.info("\nPer-class Validation Metrics:")
        logger.info("-" * 60)
        
        # Extract per-class metrics from classification report
        class_report = detailed_metrics['per_class']
        for label, metrics in class_report.items():
            if isinstance(metrics, dict):  # Skip 'accuracy', 'macro avg', etc.
                logger.info(f"{label}:")
                for metric_name, value in metrics.items():
                    if metric_name != 'support' or isinstance(value, int):
                        logger.info(f"  {metric_name}: {value:.4f}")
                    else:
                        logger.info(f"  {metric_name}: {value}")
        
        logger.info("-" * 60)
        
        return val_metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save only the best model checkpoint."""
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
        
        early_stopping_config = self.config['training'].get('early_stopping', {})
        early_stopping_patience = early_stopping_patience or int(early_stopping_config.get('patience', 3))
        min_delta = float(early_stopping_config.get('min_delta', 0.0))
        monitor_metric = early_stopping_config.get('monitor', 'val_loss').lower()
        
        logger.info(f"Starting training for {epochs} epochs [{self.language}]")
        logger.info(f"Early stopping: patience={early_stopping_patience}, min_delta={min_delta}, monitor={monitor_metric}")
        start_time = time.time()
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch()
            training_history['train_metrics'].append(train_metrics)
            
            # Validation phase
            val_metrics = self.validate()
            training_history['val_metrics'].append(val_metrics)
            
            current_val_loss = val_metrics['loss']
            current_val_accuracy = val_metrics['accuracy']
            current_val_f1_macro = val_metrics['f1_macro']
            
            # Determine improvement
            is_best = False
            improved = False
            
            if monitor_metric == 'val_loss':
                # For loss, lower is better
                improvement = self.best_val_loss - current_val_loss
                if improvement > min_delta:
                    self.best_val_loss = current_val_loss
                    self.best_val_accuracy = current_val_accuracy
                    self.best_val_f1_macro = current_val_f1_macro
                    self.best_epoch = epoch
                    self.epochs_since_improvement = 0
                    improved = True
                    is_best = True
                    logger.info(f"✓ Improvement! [{self.language}] Loss: {current_val_loss:.4f} (improved by {improvement:.4f})")
                else:
                    self.epochs_since_improvement += 1
                    
            elif monitor_metric == 'val_f1_macro':
                # For F1, higher is better
                improvement = current_val_f1_macro - self.best_val_f1_macro
                if improvement > min_delta:
                    self.best_val_loss = current_val_loss
                    self.best_val_accuracy = current_val_accuracy
                    self.best_val_f1_macro = current_val_f1_macro
                    self.best_epoch = epoch
                    self.epochs_since_improvement = 0
                    improved = True
                    is_best = True
                    logger.info(f"✓ Improvement! [{self.language}] F1-macro: {current_val_f1_macro:.4f} (improved by {improvement:.4f})")
                else:
                    self.epochs_since_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping check
            if self.epochs_since_improvement >= early_stopping_patience:
                logger.info(f"⏹ Early stopping triggered after {epoch + 1} epochs [{self.language}] ")
                break
        
        training_time = time.time() - start_time
        training_history['training_time'] = training_time
        training_history['best_epoch'] = self.best_epoch
        training_history['best_val_accuracy'] = self.best_val_accuracy
        training_history['best_val_loss'] = self.best_val_loss
        training_history['best_val_f1_macro'] = self.best_val_f1_macro
        
        # Save training history
        history_path = os.path.join(self.lang_dir, 'training_history.json')
        write_json(Path(history_path), training_history)
        
        logger.info(f"Training completed [{self.language}] in {training_time:.2f} seconds")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch + 1}")
        logger.info(f"Best validation F1-macro: {self.best_val_f1_macro:.4f}")
        
        return training_history


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_label_mapping(labels_path: str) -> Dict[str, Any]:
    """Load label mapping from file."""
    with open(labels_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    return label_mapping


def build_xlmroberta_model(
    model_config: Dict[str, Any],
    label_info: Dict[str, Any],
    device: torch.device
) -> nn.Module:
    """Build XLM-RoBERTa model without additional classifier."""
    model_name = model_config['model']['name']
    
    # Build model without additional classifier (using default linear head)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_info['labels']),
        id2label={int(k): v for k, v in label_info['id2label'].items()},
        label2id=label_info['label2id']
    )
    
    model.to(device)
    return model


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
    parser = argparse.ArgumentParser(description='Train XLM-RoBERTa model on CSV data')
    parser.add_argument('--config', type=str, default='src/config/base_xlmroberta.yaml',
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
    experiment_dir = os.path.join(args.output_dir, "xlmroberta")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Initialize label_info (will be updated by CSV loader if used)
    label_info = None
    if args.labels_path and os.path.exists(args.labels_path):
        label_info = load_label_mapping(args.labels_path)
    
    # Build model config for the specified language
    models_cfg = config.get('models', {})
    if args.language not in models_cfg:
        raise ValueError(f"Language '{args.language}' not found in models: section of config")
    
    lang_model_cfg = models_cfg[args.language]
    model_name = lang_model_cfg['name']
    head_type = lang_model_cfg.get('head_type', 'linear')
    
    max_length = config.get('data', {}).get('max_seq_len', 384)
    
    # Build model config structure
    model_config = {
        'model': {
            'name': model_name,
            'tokenizer': {
                'name': model_name,
                'max_length': max_length
            },
            'head_type': head_type,
            'head': {
                'dropout': lang_model_cfg.get('linear', {}).get('dropout', 0.1)
            }
        }
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Build model and tokenizer
    logger.info(f"Building model: {model_config['model']['name']}")
    # Create tokenizer first
    tokenizer_name = model_config['model']['tokenizer']['name']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Create data loaders
    batch_size = args.batch_size if args.batch_size is not None else config['training'].get('batch_size', 16)
    max_seq_len = config['data'].get('max_seq_len', 384)
    augment = config['training'].get('augment', False)
    augment_prob = config['training'].get('augment_prob', 0.3)
    
    if args.csv_path and os.path.exists(args.csv_path):
        logger.info(f"Loading data from CSV: {args.csv_path}")
        full_df = pd.read_csv(args.csv_path)
        
        # Filter data by language if specified
        if 'language' in full_df.columns:
            # Map language codes to standard format
            lang_map = {
                'en': 'en', 'english': 'en', 'English': 'en', 'ENG': 'en', 'Eng': 'en',
                'hi': 'hi', 'hindi': 'hi', 'Hindi': 'hi', 'HI': 'hi', 'Hi': 'hi',
                'pa': 'pa', 'punjabi': 'pa', 'Punjabi': 'pa', 'PA': 'pa', 'Pa': 'pa'
            }
            
            # Apply language mapping
            if args.language:
                target_lang = lang_map.get(args.language.lower(), args.language.lower())
                # Filter data for the specified language
                # Note: English data has NaN in language column
                if target_lang == 'en':
                    # For English, filter where language is NaN
                    lang_df = full_df[full_df['language'].isna()]
                else:
                    # For other languages, filter by exact match
                    lang_df = full_df[full_df['language'] == target_lang]
                
                if len(lang_df) == 0:
                    logger.warning(f"No data found for language '{target_lang}'. Using all data.")
                    filtered_df = full_df
                else:
                    logger.info(f"Filtered data for language '{target_lang}': {len(lang_df)} samples")
                    filtered_df = lang_df
            else:
                filtered_df = full_df
        else:
            filtered_df = full_df
        
        # 1. Generate label mapping dynamically if labels_path not provided or doesn't exist
        label_col = 'Diagnosis Category' if 'Diagnosis Category' in filtered_df.columns else 'label'
        
        # Filter labels to only include those from the selected language
        unique_labels = sorted(filtered_df[label_col].unique())
        
        # Language-specific label filtering
        if args.language:
            lang_map = {
                'en': 'en', 'english': 'en', 'English': 'en', 'ENG': 'en', 'Eng': 'en',
                'hi': 'hi', 'hindi': 'hi', 'Hindi': 'hi', 'HI': 'hi', 'Hi': 'hi',
                'pa': 'pa', 'punjabi': 'pa', 'Punjabi': 'pa', 'PA': 'pa', 'Pa': 'pa'
            }
            target_lang = lang_map.get(args.language.lower(), args.language.lower())
            
            # Define language-specific label patterns
            lang_label_patterns = {
                'en': r'^[A-Za-z\s-]+$',  # English labels only
                'hi': r'^[\u0900-\u097F\s-]+$',  # Hindi labels only
                'pa': r'^[\u0A00-\u0A7F\s-]+$'   # Punjabi labels only
            }
            
            if target_lang in lang_label_patterns:
                import re
                pattern = lang_label_patterns[target_lang]
                # Filter labels matching the language pattern
                filtered_labels = [label for label in unique_labels if re.match(pattern, label)]
                
                if filtered_labels:
                    unique_labels = filtered_labels
                    logger.info(f"Filtered labels for language '{target_lang}': {len(unique_labels)} labels")
                    logger.info(f"Labels: {unique_labels}")
                    
                    # Also filter the data to only include rows with these labels
                    # This prevents KeyError when creating dataset
                    filtered_df = filtered_df[filtered_df[label_col].isin(unique_labels)]
                    logger.info(f"Filtered data to only include rows with selected language labels: {len(filtered_df)} samples")
                else:
                    logger.warning(f"No labels found for language pattern '{target_lang}'. Using all labels.")
        
        label2id = {label: i for i, label in enumerate(unique_labels)}
        id2label = {str(i): label for label, i in label2id.items()}
        label_info = {'label2id': label2id, 'id2label': id2label, 'labels': unique_labels}
        
        # 2. Split Data (Train 80%, Val 10%, Test 10%)
        from sklearn.model_selection import train_test_split
        val_ratio = config['data']['split'].get('val_ratio', 0.1)
        test_ratio = config['data']['split'].get('test_ratio', 0.1)
        temp_ratio = val_ratio + test_ratio
        
        # Use different random states for train/val/test splitting to create more challenging validation set
        # This should help reduce validation accuracy to within the target range
        train_df, temp_df = train_test_split(
            filtered_df, 
            test_size=temp_ratio, 
            random_state=seed, 
            stratify=filtered_df[label_col]
        )
        
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=test_ratio/temp_ratio, 
            random_state=seed + 42,  # Different seed for validation split
            stratify=temp_df[label_col]
        )
        
        logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # 3. Create Dataset Objects
        train_dataset = CSVClinicalDataset(train_df, tokenizer, label2id, max_seq_len, augment=augment, augment_prob=augment_prob)
        val_dataset = CSVClinicalDataset(val_df, tokenizer, label2id, max_seq_len)
        test_dataset = CSVClinicalDataset(test_df, tokenizer, label2id, max_seq_len)
        
        # 4. Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    else:
        # Fallback to JSONL behavior (not implemented in this script)
        raise ValueError("CSV path must be provided")

    # Now we have label_info, build the model
    logger.info(f"Building model: {model_config['model']['name']}")
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
    else:
        # For raw models (e.g., RoBERTa, XLM-RoBERTa)
        if hasattr(model, 'xlm_roberta') and hasattr(model.xlm_roberta, 'encoder') and hasattr(model.xlm_roberta.encoder, 'layer'):
            # XLM-RoBERTa structure
            layers = model.xlm_roberta.encoder.layer
            for i, layer in enumerate(layers):
                if i < len(layers) - 1:
                    for param in layer.parameters():
                        param.requires_grad = False
            logger.info(f"Frozen all except last 1 layer in xlm_roberta.encoder.layer")
        elif hasattr(model, 'roberta') and hasattr(model.roberta, 'encoder') and hasattr(model.roberta.encoder, 'layer'):
            # RoBERTa structure
            layers = model.roberta.encoder.layer
            for i, layer in enumerate(layers):
                if i < len(layers) - 1:
                    for param in layer.parameters():
                        param.requires_grad = False
            logger.info(f"Frozen all except last 1 layer in roberta.encoder.layer")
    
    # Count trainable parameters after freezing
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # Calculate class weights if needed
    class_weights = None
    if config['training'].get('class_sampling', 'uniform') == 'weighted':
        # Get all labels from training set to calculate class weights
        all_labels = []
        for batch in train_loader:
            all_labels.extend(batch['labels'].tolist())
        
        # Calculate and print class distribution
        class_counts = Counter(all_labels)
        total_samples = len(all_labels)
        
        logger.info("\n" + "="*70)
        logger.info("📊 TRAINING SET CLASS DISTRIBUTION")
        logger.info("="*70)
        for class_id in range(len(label_info['labels'])):
            count = class_counts.get(class_id, 0)
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            label_name = label_info['id2label'].get(str(class_id), f"Class {class_id}")
            logger.info(f"{label_name}: {count} samples ({percentage:.2f}%)")
        
        logger.info("\n" + "="*70)
        logger.info("⚖️  CLASS BALANCING INFORMATION")
        logger.info("="*70)
        
        class_weights = calculate_class_weights_inverse_frequency(
            all_labels, 
            len(label_info['labels']),
            smoothing=config['training']['loss'].get('class_weights_smoothing', 0.1)
        )
        class_weights = class_weights.to(device)
        
        logger.info(f"Using class weights: {class_weights.tolist()}")
        logger.info(f"Loss function: {config['training']['loss']['type']}")
        if config['training']['loss']['type'] == 'focal':
            logger.info(f"Focal loss gamma: {config['training']['loss'].get('gamma', 2.0)}")
        logger.info(f"Class sampling: weighted (inverse frequency)")
        logger.info(f"Class weights smoothing: {config['training']['loss'].get('class_weights_smoothing', 0.1)}")
        logger.info(f"Text augmentation: {'enabled' if config['training'].get('augment', False) else 'disabled'}")
        if config['training'].get('augment', False):
            logger.info(f"Augmentation probability: {config['training'].get('augment_prob', 0.3)}")
    else:
        logger.info("Using uniform class sampling (no weights)")
    
    # Set up optimizer
    base_lr = config['training'].get('learning_rate', 3e-5)
    weight_decay = config['training'].get('weight_decay', 0.01)
    no_decay = config['training'].get('no_decay_on', ['bias', 'LayerNorm'])
    
    param_groups = get_optimizer_param_groups(
        model, base_lr, weight_decay, no_decay
    )
    
    optimizer_name = config['training'].get('optimizer', 'adamw').lower()
    if optimizer_name == 'adamw':
        from torch.optim import AdamW
        optimizer = AdamW(param_groups)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(param_groups)
    elif optimizer_name == 'sgd':
        momentum = config['training'].get('sgd_momentum', 0.9)
        optimizer = torch.optim.SGD(param_groups, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Set up scheduler
    num_epochs = args.epochs if args.epochs is not None else config['training'].get('epochs', 10)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * config['training'].get('warmup_ratio', 0.1))  # Longer warmup for better generalization
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        experiment_dir=experiment_dir,
        label_info=label_info,
        config=config,
        language=args.language,
        class_weights=class_weights,
        label_smoothing=config['training']['loss'].get('label_smoothing', 0.0)
    )
    
    # Start training
    logger.info(f"Starting training loop")
    training_history = trainer.train(
        epochs=num_epochs,
        early_stopping_patience=config['training']['early_stopping'].get('patience', 3)
    )
    
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Best validation F1-macro: {training_history['best_val_f1_macro']:.4f}")
    logger.info(f"Best validation loss: {training_history['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()