#!/usr/bin/env python3
"""
Baseline 2: Frozen Encoder + Nearest Centroid (NO TRAINING)
- Extract embeddings using frozen encoder ([CLS] or mean pooling)
- Compute class centroids from training data
- Predict using cosine similarity
- No gradient updates at all
"""

import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
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

class TextClassificationDataset:
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

def extract_embeddings(model, dataloader, device, pooling='cls'):
    """Extract embeddings from frozen encoder"""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get embeddings based on pooling strategy
            if pooling == 'cls':
                # Use [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0, :]
            elif pooling == 'mean':
                # Mean pooling of all tokens
                mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                embeddings = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    return np.concatenate(all_embeddings), np.concatenate(all_labels)

def compute_class_centroids(embeddings, labels):
    """Compute class centroids from embeddings"""
    unique_labels = np.unique(labels)
    centroids = {}
    
    for label in unique_labels:
        class_embeddings = embeddings[labels == label]
        centroid = np.mean(class_embeddings, axis=0)
        centroids[label] = centroid
    
    return centroids

def predict_using_centroids(embeddings, centroids):
    """Predict using cosine similarity to centroids"""
    # Convert centroids to numpy array
    centroid_labels = sorted(centroids.keys())
    centroid_array = np.array([centroids[label] for label in centroid_labels])
    
    # Compute cosine similarity between embeddings and centroids
    similarities = cosine_similarity(embeddings, centroid_array)
    
    # Get predictions (index with highest similarity)
    predictions = np.argmax(similarities, axis=1)
    
    # Map back to original labels
    predictions = np.array([centroid_labels[pred] for pred in predictions])
    
    return predictions

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

def main():
    parser = argparse.ArgumentParser(description="Frozen Encoder + Nearest Centroid Baseline")
    parser.add_argument('--model_name', type=str, required=True, choices=['indicbert', 'mdeberta'],
                        help="Model type: indicbert or mdeberta")
    parser.add_argument('--language', type=str, required=True, choices=['en', 'hi', 'pa'],
                        help="Language to train on")
    parser.add_argument('--csv_path', type=str, default='data/processed/multi_language_balanced_dataset.csv',
                        help="Path to CSV data file")
    parser.add_argument('--output_dir', type=str, default='results',
                        help="Output directory for results")
    parser.add_argument('--pooling', type=str, default='cls', choices=['cls', 'mean'],
                        help="Pooling strategy for embeddings")
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
    model = AutoModel.from_pretrained(model_mapping[args.model_name])
    
    # Create datasets
    train_dataset = TextClassificationDataset(
        data['train']['texts'], 
        data['train']['labels'], 
        tokenizer
    )
    test_dataset = TextClassificationDataset(
        data['test']['texts'], 
        data['test']['labels'], 
        tokenizer
    )
    
    # Create data loaders
    batch_size = 8
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Extract embeddings for train and test sets
    print(f"Extracting train embeddings using {args.pooling} pooling...")
    train_embeddings, train_labels = extract_embeddings(model, train_loader, device, args.pooling)
    
    print(f"Extracting test embeddings using {args.pooling} pooling...")
    test_embeddings, test_labels = extract_embeddings(model, test_loader, device, args.pooling)
    
    # Compute class centroids from train data
    print("Computing class centroids...")
    centroids = compute_class_centroids(train_embeddings, train_labels)
    
    # Predict on test set
    print("Predicting using centroids...")
    test_predictions = predict_using_centroids(test_embeddings, centroids)
    
    # Calculate metrics
    accuracy = accuracy_score(test_labels, test_predictions)
    macro_f1 = f1_score(test_labels, test_predictions, average='macro')
    balanced_acc = balanced_accuracy_score(test_labels, test_predictions)
    
    test_metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'balanced_accuracy': balanced_acc
    }
    
    print(f"Test Metrics: {test_metrics}")
    
    # Save results
    result = {
        'model': args.model_name,
        'baseline_type': 'frozen_centroid',
        'language': args.language,
        'accuracy': test_metrics['accuracy'],
        'macro_f1': test_metrics['macro_f1'],
        'balanced_accuracy': test_metrics['balanced_accuracy'],
        'num_labels': data['label_info']['num_labels'],
        'train_samples': len(data['train']['texts']),
        'test_samples': len(data['test']['texts']),
        'pooling': args.pooling
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
    detailed_file = os.path.join(args.output_dir, f'{args.model_name}_{args.language}_frozen_centroid.json')
    with open(detailed_file, 'w') as f:
        json.dump({
            'config': vars(args),
            'data_info': data['label_info'],
            'test_metrics': test_metrics
        }, f, indent=2)

if __name__ == "__main__":
    main()
