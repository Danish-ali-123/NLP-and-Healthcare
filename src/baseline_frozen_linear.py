#!/usr/bin/env python3
"""
Baseline 1: Frozen Encoder + Linear Probe
- Freeze the entire transformer backbone
- Train ONLY a single linear classification head
- Max epochs: 5
- Learning rate: 1e-3
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import os
import json
import argparse
from tqdm import tqdm
import random

# Set deterministic behavior
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class FrozenEncoderClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        # Load frozen encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Freeze all encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Add linear classification head
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        # Get [CLS] token embeddings from encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
        logits = self.classifier(cls_emb)
        return logits

def load_data(csv_path, language):
    """Load and split data by language"""
    full_df = pd.read_csv(csv_path)
    
    # Filter data by language
    if language == 'en':
        # For English, filter where language is NaN or 'en'
        lang_df = full_df[(full_df['language'].isna()) | (full_df['language'] == 'en')]
    else:
        # For other languages, filter by exact match
        lang_df = full_df[full_df['language'] == language]
    
    # Use consistent label column name
    label_col = 'Diagnosis Category' if 'Diagnosis Category' in lang_df.columns else 'label'
    
    # Create label mapping
    unique_labels = sorted(lang_df[label_col].unique())
    label2id = {str(label): i for i, label in enumerate(unique_labels)}
    id2label = {i: str(label) for label, i in label2id.items()}
    
    # Convert labels to ids
    lang_df['label_id'] = lang_df[label_col].astype(str).map(label2id)
    
    # Split data (80% train, 10% val, 10% test)
    from sklearn.model_selection import train_test_split
    
    # First split train vs temp
    train_df, temp_df = train_test_split(
        lang_df,
        test_size=0.2,
        random_state=42,
        stratify=lang_df['label_id']
    )
    
    # Then split temp into val and test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['label_id']
    )
    
    return {
        'train': {
            'texts': train_df['text_input'].tolist(),
            'labels': train_df['label_id'].tolist()
        },
        'val': {
            'texts': val_df['text_input'].tolist(),
            'labels': val_df['label_id'].tolist()
        },
        'test': {
            'texts': test_df['text_input'].tolist(),
            'labels': test_df['label_id'].tolist()
        },
        'label_info': {
            'label2id': label2id,
            'id2label': id2label,
            'num_labels': len(label2id),
            'labels': unique_labels
        }
    }

def evaluate(model, dataloader, device):
    """Evaluate model on data"""
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'balanced_accuracy': balanced_acc
    }

def train(model, train_loader, val_loader, device, num_epochs=5, lr=1e-3):
    """Train only the linear head"""
    # Only train the classifier head
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss/len(progress_bar))
        
        # Evaluate on val set
        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} Val Metrics: {val_metrics}")
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
    
    return best_val_acc

def main():
    parser = argparse.ArgumentParser(description="Frozen Encoder + Linear Probe Baseline")
    parser.add_argument('--model_name', type=str, required=True, choices=['indicbert', 'mdeberta'],
                        help="Model type: indicbert or mdeberta")
    parser.add_argument('--language', type=str, required=True, choices=['en', 'hi', 'pa'],
                        help="Language to train on")
    parser.add_argument('--csv_path', type=str, default='data/processed/multi_language_balanced_dataset.csv',
                        help="Path to CSV data file")
    parser.add_argument('--output_dir', type=str, default='results',
                        help="Output directory for results")
    args = parser.parse_args()
    
    # Set seed
    set_seed(42)
    
    # Map model name to Hugging Face model path
    model_mapping = {
        'indicbert': 'ai4bharat/indic-bert',
        'mdeberta': 'microsoft/mdeberta-v3-base'
    }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data for language: {args.language}")
    data = load_data(args.csv_path, args.language)
    
    # Load tokenizer and model
    print(f"Loading model: {model_mapping[args.model_name]}")
    tokenizer = AutoTokenizer.from_pretrained(model_mapping[args.model_name])
    
    # Create datasets
    train_dataset = TextClassificationDataset(
        data['train']['texts'], 
        data['train']['labels'], 
        tokenizer
    )
    val_dataset = TextClassificationDataset(
        data['val']['texts'], 
        data['val']['labels'], 
        tokenizer
    )
    test_dataset = TextClassificationDataset(
        data['test']['texts'], 
        data['test']['labels'], 
        tokenizer
    )
    
    # Create data loaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with frozen encoder
    model = FrozenEncoderClassifier(
        model_mapping[args.model_name],
        num_labels=data['label_info']['num_labels']
    )
    model.to(device)
    
    # Verify encoder is frozen
    encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    classifier_params = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    print(f"Trainable parameters - Encoder: {encoder_params}, Classifier: {classifier_params}")
    
    # Train only the linear head
    print("Training linear classification head...")
    train(model, train_loader, val_loader, device, num_epochs=5, lr=1e-3)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test Metrics: {test_metrics}")
    
    # Save results
    result = {
        'model': args.model_name,
        'baseline_type': 'frozen_linear',
        'language': args.language,
        'accuracy': test_metrics['accuracy'],
        'macro_f1': test_metrics['macro_f1'],
        'balanced_accuracy': test_metrics['balanced_accuracy'],
        'num_labels': data['label_info']['num_labels'],
        'train_samples': len(data['train']['texts']),
        'test_samples': len(data['test']['texts'])
    }
    
    # Append to results CSV
    results_file = os.path.join(args.output_dir, 'baseline_results.csv')
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
        results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)
    else:
        results_df = pd.DataFrame([result])
    
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")
    
    # Save detailed results for debugging
    detailed_file = os.path.join(args.output_dir, f'{args.model_name}_{args.language}_frozen_linear.json')
    with open(detailed_file, 'w') as f:
        json.dump({
            'config': vars(args),
            'data_info': data['label_info'],
            'test_metrics': test_metrics
        }, f, indent=2)

if __name__ == "__main__":
    main()
