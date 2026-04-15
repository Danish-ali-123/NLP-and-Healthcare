#!/usr/bin/env python3
"""
Real-time prediction script for IndicBERT-HPA models trained on JSONL data.

This script loads a trained model from checkpoint and performs real-time predictions
on input text, with optional metrics calculation if ground truth labels are provided.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import Dict, Any, List, Tuple, Optional
import logging
import os
import json
import argparse
import yaml
import numpy as np
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# Transliteration support
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Add src to path for imports
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Local imports
from src.models.metrics import MetricCalculator
from src.models.indicbert_hpa import IndicBertHPAClassifier
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
        self.max_seq_len = self.config.get('data', {}).get('max_seq_len', 320)
        
        # Build model config
        self.model_config = self._build_model_config(self.config, self.config.get('language', 'en'))
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer(checkpoint_path)
        
        # Initialize metric calculator
        self.metric_calculator = MetricCalculator(self.id2label)
        
        # Store selected language (will be set in interactive mode)
        self.selected_lang = None
        
    def _load_label_mapping(self, labels_path: str) -> Dict[str, Any]:
        """Load label mapping from file."""
        with open(labels_path, 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
        return label_mapping
    
    def _build_model_config(self, config: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Build model configuration from base config."""
        # Check if config has 'models' section (for multi-language support)
        if 'models' in config:
            models_cfg = config['models']
            if language not in models_cfg:
                raise ValueError(f"Language '{language}' not found in models: section of config")
            lang_model_cfg = models_cfg[language]
        elif 'model' in config:
            # For single-model configs (like IndicBERT-HPA)
            lang_model_cfg = config['model']
        else:
            raise ValueError("Config must have either 'models' or 'model' section")
        
        model_name = lang_model_cfg['name']
        head_type = lang_model_cfg.get('head_type', 'linear')
        
        # Build model config structure
        model_config = {
            'model': {
                'name': model_name,
                'tokenizer': {
                    'name': model_name,
                    'max_length': self.max_seq_len
                },
                'head_type': head_type
            }
        }
        
        # Add HPA configuration if present (for IndicBERT-HPA models)
        if 'hpa' in lang_model_cfg:
            model_config['model']['hpa'] = lang_model_cfg['hpa']
        
        return model_config
    
    def _load_model_and_tokenizer(self, checkpoint_path: str) -> Tuple[nn.Module, Any]:
        """Load model and tokenizer from checkpoint."""
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        
        # Check if checkpoint is a file or directory
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model_state_dict = checkpoint['model_state_dict']
        else:
            # Assume it's a directory with best.ckpt, possibly in language subdirectory
            checkpoint_file = os.path.join(checkpoint_path, 'best.ckpt')
            
            # If we have a selected language, try to load from that language's directory first
            if hasattr(self, 'selected_lang') and self.selected_lang:
                lang_checkpoint_file = os.path.join(checkpoint_path, self.selected_lang, 'best.ckpt')
                if os.path.exists(lang_checkpoint_file):
                    checkpoint_file = lang_checkpoint_file
                    logger.info(f"Loading checkpoint from language-specific directory: {self.selected_lang}")
            
            # If no language-specific checkpoint found or no language selected, try other options
            if not os.path.exists(checkpoint_file):
                # Check in language subdirectory (like 'en')
                lang_dirs = [d for d in os.listdir(checkpoint_path) if os.path.isdir(os.path.join(checkpoint_path, d))]
                for lang_dir in lang_dirs:
                    lang_checkpoint_file = os.path.join(checkpoint_path, lang_dir, 'best.ckpt')
                    if os.path.exists(lang_checkpoint_file):
                        checkpoint_file = lang_checkpoint_file
                        logger.info(f"Loading checkpoint from language directory: {lang_dir}")
                        break
                if not os.path.exists(checkpoint_file):
                    raise FileNotFoundError(f"Best checkpoint file not found: {checkpoint_file}")
            
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            model_state_dict = checkpoint['model_state_dict']
        
        # Get model name from config
        model_name = self.model_config['model']['name']
        
        # Get HPA configuration
        hpa_config = self.model_config.get('model', {}).get('hpa', {})
        adapter_hidden_sizes = hpa_config.get('hidden_sizes', [512])
        adapter_dropout = hpa_config.get('dropout', 0.2)
        
        # Build IndicBERT-HPA model
        model = IndicBertHPAClassifier(
            backbone_checkpoint=model_name,
            num_labels=len(self.label_info['labels']),
            id2label={int(k): v for k, v in self.label_info['id2label'].items()},
            label2id=self.label2id,
            adapter_hidden_sizes=adapter_hidden_sizes,
            adapter_dropout=adapter_dropout
        )
        
        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load state dict
        model.load_state_dict(model_state_dict)
        model.to(self.device)
        model.eval()
        
        logger.info(f"Successfully loaded IndicBERT-HPA model: {model_name}")
        logger.info(f"Adapter architecture: {adapter_hidden_sizes} with dropout={adapter_dropout}")
        return model, tokenizer
    
    def predict_single(self, text: str, age: int = 30, gender: str = "patient") -> Dict[str, Any]:
        """
        Predict class for a single text input.
        
        Args:
            text: Input text to classify
            age: Age of the patient (default: 30)
            gender: Gender of the patient (default: "patient")
            
        Returns:
            Dict with prediction results
        """
        # Apply transliteration if needed based on selected language
        processed_text = text
        
        if hasattr(self, 'selected_lang') and self.selected_lang:
            if self.selected_lang == 'hi':
                # Transliterate Roman Hindi (Hinglish) to Devanagari
                try:
                    processed_text = transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
                except Exception as e:
                    # If transliteration fails, use original text
                    pass
            elif self.selected_lang == 'pa':
                # Transliterate Roman Punjabi to Gurmukhi
                try:
                    processed_text = transliterate(text, sanscript.ITRANS, sanscript.GURMUKHI)
                except Exception as e:
                    # If transliteration fails, use original text
                    pass
        
        # Apply clinical masking (same as training script)
        masked_text = re.sub(r'hip|spine|spinal|back|groin|lower joint area|mid-body axis|upper extremity joint', '[LOCATION]', processed_text, flags=re.IGNORECASE)
        
        # Format the text the same way as training data (text_input format)
        formatted_text = masked_text
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove token_type_ids if present as some models (like DistilBERT, IndicBERT) don't use them
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
        
        prediction_label = self.id2label[pred_id]
        probability = probs[0][pred_id].item()
        
        # Create result dict
        result = {
            'text': text,
            'prediction': {
                'label': prediction_label,
                'id': pred_id,
                'probability': float(probability),
                'probabilities': {
                    self.id2label[i]: float(p) for i, p in enumerate(probs[0].cpu().numpy())
                },
                'logits': {
                    self.id2label[i]: float(l) for i, l in enumerate(outputs.logits[0].cpu().numpy())
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
        # Apply clinical masking (same as training script) to all texts
        masked_texts = []
        for text in texts:
            masked = re.sub(r'hip|spine|spinal|back|groin|lower joint area|mid-body axis|upper extremity joint', '[LOCATION]', text, flags=re.IGNORECASE)
            masked_texts.append(masked)
        
        # Tokenize inputs
        inputs = self.tokenizer(
            masked_texts,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove token_type_ids if present as some models (like DistilBERT, IndicBERT) don't use them
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
        
        # Convert to numpy
        probs_np = probs.cpu().numpy()
        predictions_idx = predictions.cpu().numpy()
        
        # Create results
        results = []
        for i, text in enumerate(texts):
            prediction_idx = predictions_idx[i]
            prediction_label = self.id2label[prediction_idx]
            
            result = {
                'text': text,
                'prediction': {
                    'label': prediction_label,
                    'id': prediction_idx,
                    'probability': float(probs_np[i][prediction_idx]),
                    'probabilities': {
                        self.id2label[j]: float(p) for j, p in enumerate(probs_np[i])
                    },
                    'logits': {
                        self.id2label[j]: float(l) for j, l in enumerate(outputs.logits[i].cpu().numpy())
                    }
                }
            }
            results.append(result)
        
        return results
    
    def predict_with_metrics(self, texts: List[str], true_labels: List[str]) -> Dict[str, Any]:
        """
        Predict classes and calculate metrics.
        
        Args:
            texts: List of input texts to classify
            true_labels: List of ground truth labels
            
        Returns:
            Dict with predictions and metrics
        """
        # Predict batch with clinical masking
        predictions = self.predict_batch(texts)
        
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
        print("🚀 Real-time IndicBERT-HPA Prediction for Clinical Data")
        print("="*70)
        print(f"Model: {self.model_config['model']['name']}")
        print(f"Labels: {', '.join(self.id2label.values())}")
        print(f"Max sequence length: {self.max_seq_len}")
        print("="*70)
        print("Type 'exit' to quit.")
        print("="*70)
        
        # Language selection
        print("\n🌐 LANGUAGE SELECTION")
        print("------------------------")
        print("Please select the language for input:")
        print("  - English (en, English, ENG, etc.)")
        print("  - Hindi (hi, Hindi, HI, etc.)")
        print("  - Punjabi (pa, Punjabi, PA, etc.)")
        
        while True:
            lang_input = input("\nEnter language: ").strip()
            
            if lang_input.lower() == 'exit':
                print("\n👋 Exiting...")
                return
            
            # Map language input to standard code
            lang_map = {
                'en': 'en', 'english': 'en', 'English': 'en', 'ENG': 'en', 'Eng': 'en', 'eng': 'en',
                'hi': 'hi', 'hindi': 'hi', 'Hindi': 'hi', 'HI': 'hi', 'Hi': 'hi', 'hi': 'hi',
                'pa': 'pa', 'punjabi': 'pa', 'Punjabi': 'pa', 'PA': 'pa', 'Pa': 'pa', 'pa': 'pa'
            }
            
            selected_lang = lang_map.get(lang_input.lower())
            
            if not selected_lang:
                print("⚠️  Invalid language. Please try again.")
                continue
            
            # Get language name for display
            lang_names = {'en': 'English', 'hi': 'Hindi', 'pa': 'Punjabi'}
            lang_name = lang_names.get(selected_lang, selected_lang)
            
            print(f"\n✅ Selected language: {lang_name}")
            # Store selected language in instance variable
            self.selected_lang = selected_lang
            break
        
        # Text input based on selected language
        print(f"\n📝 Enter text in {lang_name}:")
        print("Note: For Hindi and Punjabi, romanized input is supported.")
        print("="*70)
        
        # Language-specific validation patterns
        lang_patterns = {
            'en': r'^[a-zA-Z0-9\s.,!?-]+$',  # English (basic)
            'hi': r'^[\u0900-\u097F\sa-zA-Z0-9.,!?-]+$',  # Hindi (Devanagari) + Roman
            'pa': r'^[\u0A00-\u0A7F\sa-zA-Z0-9.,!?-]+$'   # Punjabi (Gurmukhi) + Roman
        }
        
        while True:
            try:
                # Get input text
                text = input("\nEnter text for prediction: ").strip()
                
                if text.lower() == 'exit':
                    print("\n👋 Exiting...")
                    break
                
                if not text.strip():
                    print("⚠️  Please enter some text.")
                    continue
                
                # Language-specific input validation
                if selected_lang in lang_patterns:
                    import re
                    pattern = lang_patterns[selected_lang]
                    if not re.match(pattern, text):
                        print(f"⚠️  Please enter text in {lang_name}.")
                        continue
                
                # Make prediction
                result = self.predict_single(text)
                
                # Display results
                self._display_prediction_result(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def _display_prediction_result(self, result: Dict[str, Any]):
        """Display prediction result in a user-friendly format."""
        print("\n" + "="*50)
        print("📝 TEXT")
        print("="*50)
        print(result['text'])
        print("\n" + "="*50)
        print("🎯 PREDICTION RESULT")
        print("="*50)
        
        pred = result['prediction']
        print(f"Predicted Label: {pred['label']} (ID: {pred['id']})")
        print(f"Confidence: {pred['probability']:.4f}")
        
        print("\n📊 CLASS PROBABILITIES")
        print("-"*30)
        # Sort probabilities in descending order
        sorted_probs = sorted(pred['probabilities'].items(), key=lambda x: x[1], reverse=True)
        for label, prob in sorted_probs:
            print(f"{label}: {prob:.4f}")
        print("="*50)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Real-time prediction with trained IndicBERT-HPA model')
    
    parser.add_argument('--config', type=str, default='src/config/base_indicbert_hpa.yaml',
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
    
    # Initialize predictor
    predictor = RealTimePredictor(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config,
        labels_path=args.labels_path
    )
    
    # Run in appropriate mode
    if args.interactive:
        # Interactive mode
        predictor.interactive_predict()
    elif args.text:
        # Single text prediction
        result = predictor.predict_single(args.text)
        predictor._display_prediction_result(result)
    elif args.texts_file:
        # Batch prediction
        with open(args.texts_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        if args.labels_file:
            # With metrics (ground truth labels provided)
            with open(args.labels_file, 'r', encoding='utf-8') as f:
                true_labels = [line.strip() for line in f if line.strip()]
            
            if len(texts) != len(true_labels):
                print(f"❌ Error: Number of texts ({len(texts)}) doesn't match number of labels ({len(true_labels)})")
                return
            
            result = predictor.predict_with_metrics(texts, true_labels)
            
            # Display metrics
            print("\n" + "="*60)
            print("📋 METRICS SUMMARY")
            print("="*60)
            
            # Print basic metrics
            print("Basic Metrics:")
            print("-"*30)
            for metric, value in result['metrics']['basic_metrics'].items():
                if metric != 'loss':
                    print(f"{metric}: {value:.4f}")
            
            print("\nClassification Report:")
            print("-"*30)
            # Print classification report in a readable format
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
            # Without metrics (just predictions)
            results = predictor.predict_batch(texts)
            
            print("\n" + "="*60)
            print("📋 BATCH PREDICTION RESULTS")
            print("="*60)
            
            for i, result in enumerate(results):
                print(f"\nPrediction {i+1}:")
                print(f"Text: {result['text'][:100]}..." if len(result['text']) > 100 else f"Text: {result['text']}")
                print(f"Predicted: {result['prediction']['label']} (Confidence: {result['prediction']['probability']:.4f})")
    else:
        # No mode specified - show help
        parser.print_help()


if __name__ == "__main__":
    main()