#!/usr/bin/env python3
"""
Real-time prediction script for XLM-RoBERTa models trained on JSONL data.

This script loads a trained model from checkpoint and performs real-time predictions
on input text, with optional metrics calculation if ground truth labels are provided.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
import logging
import os
import json
import argparse
import yaml
import numpy as np
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# Add src to path for imports
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Local imports
from src.models.metrics import MetricCalculator
from src.utils.io import read_json
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


class RealTimePredictor:
    """Real-time predictor for clinical text classification models."""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        labels_path: str,
        device: Optional[str] = None
    ):
        """
        Initialize the real-time predictor.
        
        Args:
            checkpoint_path: Path to model checkpoint file or directory
            config_path: Path to configuration file
            labels_path: Path to labels mapping file
            device: Device to run inference on (cuda/cpu)
        """
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Set device
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        logger.info(f"Using device: {self.device}")
        
        # Load configuration
        self.config = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
        
        # Set seed for reproducibility
        seed = self.config.get('seed', 42)
        set_seed(seed)
        
        # Load label mapping
        self.label_info = self._load_label_mapping(labels_path)
        self.id2label = {int(k): v for k, v in self.label_info['id2label'].items()}
        self.label2id = {v: int(k) for k, v in self.label_info['id2label'].items()}
        
        # Get max sequence length from config
        self.max_seq_len = self.config.get('data', {}).get('max_seq_len', 256)
        
        # Build model config
        self.model_config = self._build_model_config(self.config, self.config.get('language', 'en'))
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer(checkpoint_path)
        
        # Initialize metric calculator
        self.metric_calculator = MetricCalculator(self.id2label)
        
    def _load_label_mapping(self, labels_path: str) -> Dict[str, Any]:
        """Load label mapping from file."""
        with open(labels_path, 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
        return label_mapping
    
    def _build_model_config(self, config: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Build model configuration from base config."""
        models_cfg = config.get('models', {})
        if language not in models_cfg:
            raise ValueError(f"Language '{language}' not found in models section")
        
        lang_model_cfg = models_cfg[language]
        model_name = lang_model_cfg['name']
        head_type = lang_model_cfg.get('head_type', 'linear')
        
        return {
            'model': {
                'name': model_name,
                'tokenizer': {
                    'name': model_name,
                    'max_length': self.max_seq_len
                },
                'head_type': head_type
            }
        }
    
    def _load_model_and_tokenizer(self, checkpoint_path: str) -> Tuple[nn.Module, Any]:
        """Load model from checkpoint."""
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        
        # Normalize the path to handle different formats
        checkpoint_path = os.path.normpath(checkpoint_path)
        
        # Check if checkpoint is a file (ends with .ckpt) or directory
        if checkpoint_path.endswith('.ckpt'):
            # Direct path to checkpoint file
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model_state_dict = checkpoint['model_state_dict']
        else:
            # Assume it's a directory with best.ckpt
            checkpoint_file = os.path.join(checkpoint_path, 'best.ckpt')
            if not os.path.exists(checkpoint_file):
                raise FileNotFoundError(f"Best checkpoint file not found: {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            model_state_dict = checkpoint['model_state_dict']
        
        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_config['model']['name'])
        
        # Build model the same way it was trained (using AutoModelForSequenceClassification)
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_config['model']['name'],
            num_labels=len(self.label_info['labels']),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Load state dict
        model.load_state_dict(model_state_dict)
        model.to(self.device)
        model.eval()
        
        logger.info(f"Successfully loaded model: {self.model_config['model']['name']}")
        return model, tokenizer
    def clean_clinical_text(self, text):
        replacements = {
            r'hip|groin|thigh': 'lower joint area',
            r'spinal|spine|back|vertebral': 'mid-body axis',
            r'shoulder|elbow|wrist': 'upper extremity joint',
            r'knee|ankle|foot': 'lower extremity joint'
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', text).strip()
    
    def predict_single(self, text: str, age: int = 30, gender: str = "patient", history: str = "no previous history",remarks: str = "none") -> Dict[str, Any]:
        """
        Predict class for a single text input.
        
        Args:
            text: Input text to classify (can be raw text or pre-formatted text_input)
            age: Age of the patient (default: 30) - used if constructing format
            gender: Gender of the patient (default: "patient") - used if constructing format
            
        Returns:
            Dict with prediction results
        """
        linearized_text = (f"Age: {float(age)}, Gender: {gender}. "
                       f"Symptoms: {text}. "
                       f"History: {history}. "
                       f"Remarks: {remarks}.")
        linearized_text = self.clean_clinical_text(linearized_text)
        
        # Apply the exact same location masking as in training
        linearized_text = re.sub(r'hip|spine|spinal|back|groin|lower joint area|mid-body axis|upper extremity joint', '[LOCATION]', linearized_text, flags=re.IGNORECASE)
        
        # Tokenize with exact same parameters as training
        inputs = self.tokenizer(
            linearized_text,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=-1)
        
        logits_np = logits.cpu().numpy()[0]
        probabilities_np = probabilities.cpu().numpy()[0]
        prediction_idx = predictions.cpu().item()
        prediction_label = self.id2label[prediction_idx]
        
        result = {
            'text': text,
            'formatted_text': linearized_text,
            'prediction': {
                'label': prediction_label,
                'id': prediction_idx,
                'probability': float(probabilities_np[prediction_idx]),
                'probabilities': {
                    self.id2label[i]: float(p) for i, p in enumerate(probabilities_np)
                },
                'logits': {
                    self.id2label[i]: float(l) for i, l in enumerate(logits_np)
                }
            }
        }
        
        return result
    
    def predict_batch(self, texts: List[str], ages: Optional[List[int]] = None, genders: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Predict classes for multiple text inputs.
        
        Args:
            texts: List of input texts to classify
            ages: List of ages for each patient (default: 30 for all)
            genders: List of genders for each patient (default: "patient" for all)
            
        Returns:
            List of dicts with prediction results
        """
        if ages is None:
            ages = [30] * len(texts)
        if genders is None:
            genders = ["patient"] * len(texts)
        
        linearized_texts = []
        for text, age, gender in zip(texts, ages, genders):
            if "Patient Age:" in text and "Gender:" in text and "Symptoms:" in text and "History:" in text:
                linearized_text = text
            else:
                linearized_text = f"Patient Age: {age}, Gender: {gender}. Symptoms: {text}."
            
            # Apply the exact same location masking as in training
            linearized_text = re.sub(r'hip|spine|spinal|back|groin', '[LOCATION]', linearized_text, flags=re.IGNORECASE)
            linearized_texts.append(linearized_text)
        
        # Tokenize with exact same parameters as training
        inputs = self.tokenizer(
            linearized_texts,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=-1)
        
        logits_np = logits.cpu().numpy()
        probabilities_np = probabilities.cpu().numpy()
        predictions_idx = predictions.cpu().numpy()
        
        results = []
        for i, text in enumerate(texts):
            prediction_idx = predictions_idx[i]
            prediction_label = self.id2label[prediction_idx]
            
            result = {
                'text': text,
                'formatted_text': linearized_texts[i],
                'prediction': {
                    'label': prediction_label,
                    'id': prediction_idx,
                    'probability': float(probabilities_np[i][prediction_idx]),
                    'probabilities': {
                        self.id2label[j]: float(p) for j, p in enumerate(probabilities_np[i])
                    },
                    'logits': {
                        self.id2label[j]: float(l) for j, l in enumerate(logits_np[i])
                    }
                }
            }
            results.append(result)
        
        return results
    
    def predict_with_metrics(self, texts: List[str], true_labels: List[str], ages: Optional[List[int]] = None, genders: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Predict classes and calculate metrics.
        
        Args:
            texts: List of input texts to classify
            true_labels: List of ground truth labels
            ages: List of ages for each patient (default: 30 for all)
            genders: List of genders for each patient (default: "patient" for all)
            
        Returns:
            Dict with predictions and metrics
        """
        # Predict batch with linearization
        predictions = self.predict_batch(texts, ages, genders)
        
        # Extract predicted labels
        pred_labels = [p['prediction']['label'] for p in predictions]
        pred_ids = [p['prediction']['id'] for p in predictions]
        
        # Convert true labels to ids
        true_ids = [self.label2id[label] for label in true_labels]
        
        # Calculate metrics
        metrics = self.metric_calculator.compute_metrics((pred_ids, true_ids))
        
        # Calculate classification report
        class_report = classification_report(
            true_ids, pred_ids,
            labels=list(range(len(self.id2label))),
            target_names=list(self.id2label.values()),
            output_dict=True,
            zero_division=0
        )
        
        return {
            'predictions': predictions,
            'metrics': {
                'basic_metrics': metrics,
                'classification_report': class_report
            }
        }
    
    def interactive_predict(self):
        """Interactive prediction mode."""
        print("\n" + "="*70)
        print(" Real-time XLM-RoBERTa Prediction for Clinical Data")
        print("="*70)
        print(f"Model: {self.model_config['model']['name']}")
        print(f"Labels: {', '.join(self.id2label.values())}")
        print(f"Max sequence length: {self.max_seq_len}")
        print("="*70)
        print("Type 'exit' to quit.")
        print("="*70)
        
        while True:
            try:
                text = input("\nEnter text for prediction: ")
                
                if text.lower() == 'exit':
                    print("\n Exiting...")
                    break
                
                if not text.strip():
                    print(" Please enter some text.")
                    continue
                
                result = self.predict_single(text)
                
                self._display_prediction_result(result)
                
            except KeyboardInterrupt:
                print("\n\n Exiting...")
                break
            except Exception as e:
                print(f" Error: {str(e)}")
    
    def _display_prediction_result(self, result: Dict[str, Any]):
        """Display prediction result in a user-friendly format."""
        print("\n" + "="*50)
        print("TEXT")
        print("="*50)
        print(result['text'])
        print("\n" + "="*50)
        print("PREDICTION RESULT")
        print("="*50)
        
        pred = result['prediction']
        print(f"Predicted Label: {pred['label']} (ID: {pred['id']})")
        print(f"Confidence: {pred['probability']:.4f}")
        
        print("\nCLASS PROBABILITIES")
        print("-"*30)
        sorted_probs = sorted(pred['probabilities'].items(), key=lambda x: x[1], reverse=True)
        for label, prob in sorted_probs:
            print(f"{label}: {prob:.4f}")
        print("="*50)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Real-time prediction with trained XLM-RoBERTa model')
    
    parser.add_argument('--config', type=str, default='src/config/base_xlmroberta.yaml',
                        help='Path to configuration file')
    parser.add_argument('--labels_path', type=str, default='preprocess_jsonl/labels.json',
                        help='Path to labels JSON file')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint file or directory')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--text', type=str, default=None,
                        help='Single text to predict (non-interactive mode)')
    parser.add_argument('--texts_file', type=str, default=None,
                        help='Path to file with texts for batch prediction')
    parser.add_argument('--labels_file', type=str, default=None,
                        help='Path to file with ground truth labels (for metrics)')
    
    args = parser.parse_args()
    
    predictor = RealTimePredictor(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config,
        labels_path=args.labels_path
    )
    
    if args.interactive:
        predictor.interactive_predict()
    elif args.text:
        result = predictor.predict_single(args.text)
        predictor._display_prediction_result(result)
    elif args.texts_file:
        with open(args.texts_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        if args.labels_file:
            with open(args.labels_file, 'r', encoding='utf-8') as f:
                true_labels = [line.strip() for line in f if line.strip()]
            
            if len(texts) != len(true_labels):
                print(f"Error: Number of texts ({len(texts)}) doesn't match number of labels ({len(true_labels)})")
                return
            
            result = predictor.predict_with_metrics(texts, true_labels)
            
            print("\n" + "="*60)
            print("METRICS SUMMARY")
            print("="*60)
            
            print("Basic Metrics:")
            print("-"*30)
            for metric, value in result['metrics']['basic_metrics'].items():
                if metric != 'loss':
                    print(f"{metric}: {value:.4f}")
            
            print("\nClassification Report:")
            print("-"*30)
            report = result['metrics']['classification_report']
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    print(f"\n{label}:")
                    for metric, value in metrics.items():
                        if metric not in ['support'] or isinstance(value, int):
                            print(f"  {metric}: {value:.4f}")
                        else:
                            print(f"  {metric}: {value}")
        else:
            results = predictor.predict_batch(texts)
            
            print("\n" + "="*60)
            print("BATCH PREDICTION RESULTS")
            print("="*60)
            
            for i, result in enumerate(results):
                print(f"\nPrediction {i+1}:")
                print(f"Text: {result['text'][:100]}..." if len(result['text']) > 100 else f"Text: {result['text']}")
                print(f"Predicted: {result['prediction']['label']} (Confidence: {result['prediction']['probability']:.4f})")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()