#!/usr/bin/env python3
"""
Real-time prediction script for DistilBERT models trained on JSONL/CSV data.

Supports:
- Fine-tuned mode (default): loads best.ckpt (or explicit .ckpt file)
- Baseline mode (--no_ckpt): uses pretrained encoder + random/untrained head
  and (optionally) re-initializes the head deterministically via --seed.

Usage examples:
  # Fine-tuned (load checkpoint directory containing best.ckpt)
  python predict_jsonl_distilbert.py --checkpoint_path train_jsonl/experiments/distilbert/en

  # Baseline (no checkpoint)
  python predict_jsonl_distilbert.py --checkpoint_path dummy --no_ckpt --seed 123 --text "pain in back"

Note: In baseline mode, checkpoint_path is ignored but kept for interface compatibility.
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
from sklearn.metrics import classification_report

# Add src to path for imports
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Local imports
from src.models.metrics import MetricCalculator
from src.models.baselines import build_model_and_tokenizer
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def _reset_module_parameters(module: nn.Module) -> None:
    """
    Reset parameters of a module if possible.
    This is used to re-randomize the classification head in baseline mode.
    """
    if module is None:
        return
    # Common PyTorch modules have reset_parameters()
    if hasattr(module, "reset_parameters") and callable(getattr(module, "reset_parameters")):
        module.reset_parameters()
        return
    # Otherwise, try to reset submodules
    for m in module.modules():
        if hasattr(m, "reset_parameters") and callable(getattr(m, "reset_parameters")):
            try:
                m.reset_parameters()
            except Exception:
                pass


def _find_and_reset_classifier_head(model: nn.Module) -> bool:
    """
    Try to find the classifier head on the model and reset it.
    Returns True if a head was found and reset, else False.
    """
    candidate_attrs = [
        "classifier",
        "head",
        "classification_head",
        "fc",
        "output_layer",
        "proj",
    ]

    # First try direct attributes
    for attr in candidate_attrs:
        if hasattr(model, attr):
            head = getattr(model, attr)
            if isinstance(head, nn.Module):
                _reset_module_parameters(head)
                return True

    # Try nested common patterns
    # e.g., model.model.classifier (HuggingFace style wrappers)
    if hasattr(model, "model") and isinstance(getattr(model, "model"), nn.Module):
        inner = getattr(model, "model")
        for attr in candidate_attrs:
            if hasattr(inner, attr):
                head = getattr(inner, attr)
                if isinstance(head, nn.Module):
                    _reset_module_parameters(head)
                    return True

    return False


class RealTimePredictor:
    """Real-time predictor for clinical text classification models."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        labels_path: str,
        device: Optional[str] = None,
        no_ckpt: bool = False,
        seed: Optional[int] = None
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint file/dir (ignored if no_ckpt=True)
            config_path: YAML config
            labels_path: labels.json containing id2label/label2id
            device: cuda/cpu
            no_ckpt: If True, do NOT load checkpoint -> baseline mode
            seed: If provided, set seed AND re-init classifier head for baseline reproducibility
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

        self.no_ckpt = no_ckpt

        # Device
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Using device: {self.device}")

        # Load configuration
        self.config = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

        # Set seed (important for baseline head init reproducibility)
        # Priority: explicit seed arg > config seed > default 42
        final_seed = seed if seed is not None else int(self.config.get("seed", 42))
        set_seed(final_seed)
        self.seed = final_seed
        logger.info(f"Seed set to: {self.seed}")

        # Load labels
        with open(labels_path, "r", encoding="utf-8") as f:
            self.label_info = json.load(f)

        self.id2label = {int(k): v for k, v in self.label_info["id2label"].items()}
        self.label2id = {v: int(k) for k, v in self.label_info["id2label"].items()}

        # Max length
        self.max_seq_len = self.config.get("data", {}).get("max_seq_len", 384)

        # Build model_config from config language section
        self.model_config = self._build_model_config(self.config, self.config.get("language", "en"))

        # Load model/tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer(checkpoint_path)

        # Metrics helper
        self.metric_calculator = MetricCalculator(self.id2label)

    def _build_model_config(self, config: Dict[str, Any], language: str) -> Dict[str, Any]:
        models_cfg = config.get("models", {})
        if language not in models_cfg:
            raise ValueError(f"Language '{language}' not found in models section")

        lang_model_cfg = models_cfg[language]
        model_name = lang_model_cfg["name"]
        head_type = lang_model_cfg.get("head_type", "linear")

        return {
            "model": {
                "name": model_name,
                "tokenizer": {
                    "name": model_name,
                    "max_length": self.max_seq_len
                },
                "head_type": head_type
            }
        }

    def _load_model_and_tokenizer(self, checkpoint_path: str) -> Tuple[nn.Module, Any]:
        """
        Fine-tuned mode: load checkpoint state_dict.
        Baseline mode (--no_ckpt): skip checkpoint loading and reset classifier head.
        """
        # Build model and tokenizer first
        model, tokenizer = build_model_and_tokenizer(
            model_config=self.model_config,
            label_info=self.label_info,
            device=self.device
        )

        if self.no_ckpt:
            logger.info("✅ Baseline mode enabled (--no_ckpt): skipping checkpoint loading.")
            logger.info("Re-initializing classifier head weights for a true baseline.")
            ok = _find_and_reset_classifier_head(model)
            if not ok:
                logger.warning(
                    "⚠️ Could not find a known classifier-head attribute to reset. "
                    "Baseline may still behave deterministically. "
                    "If your model uses a custom head name, tell me that attribute."
                )
            model.to(self.device)
            model.eval()
            logger.info(f"Baseline model ready: {self.model_config['model']['name']}")
            return model, tokenizer

        # Fine-tuned mode: load checkpoint
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        checkpoint_path = os.path.normpath(checkpoint_path)

        if checkpoint_path.endswith(".ckpt"):
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model_state_dict = checkpoint["model_state_dict"]
        else:
            checkpoint_file = os.path.join(checkpoint_path, "best.ckpt")
            if not os.path.exists(checkpoint_file):
                raise FileNotFoundError(f"Best checkpoint file not found: {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            model_state_dict = checkpoint["model_state_dict"]

        model.load_state_dict(model_state_dict, strict=True)
        model.to(self.device)
        model.eval()
        logger.info(f"✅ Successfully loaded fine-tuned model: {self.model_config['model']['name']}")
        return model, tokenizer

    def clean_clinical_text(self, text: str) -> str:
        replacements = {
            r"hip|groin|thigh": "lower joint area",
            r"spinal|spine|back|vertebral": "mid-body axis",
            r"shoulder|elbow|wrist": "upper extremity joint",
            r"knee|ankle|foot": "lower extremity joint",
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip()

    def predict_single(
        self,
        text: str,
        age: int = 30,
        gender: str = "patient",
        history: str = "no previous history",
        remarks: str = "none",
    ) -> Dict[str, Any]:
        linearized_text = (
            f"Age: {float(age)}, Gender: {gender}. "
            f"Symptoms: {text}. "
            f"History: {history}. "
            f"Remarks: {remarks}."
        )
        linearized_text = self.clean_clinical_text(linearized_text)

        # Mask locations (matches training)
        linearized_text = re.sub(
            r"hip|spine|spinal|back|groin|lower joint area|mid-body axis|upper extremity joint",
            "[LOCATION]",
            linearized_text,
            flags=re.IGNORECASE,
        )

        inputs = self.tokenizer(
            linearized_text,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            pred_id = torch.argmax(logits, dim=-1).item()

        logits_np = logits.detach().cpu().numpy()[0]
        probs_np = probabilities.detach().cpu().numpy()[0]
        pred_label = self.id2label[pred_id]

        return {
            "text": text,
            "formatted_text": linearized_text,
            "prediction": {
                "label": pred_label,
                "id": pred_id,
                "probability": float(probs_np[pred_id]),
                "probabilities": {self.id2label[i]: float(p) for i, p in enumerate(probs_np)},
                "logits": {self.id2label[i]: float(l) for i, l in enumerate(logits_np)},
            },
        }

    def predict_batch(
        self,
        texts: List[str],
        ages: Optional[List[int]] = None,
        genders: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if ages is None:
            ages = [30] * len(texts)
        if genders is None:
            genders = ["patient"] * len(texts)

        linearized_texts = []
        for text, age, gender in zip(texts, ages, genders):
            linearized_text = f"Patient Age: {age}, Gender: {gender}. Symptoms: {text}."
            linearized_text = re.sub(r"hip|spine|spinal|back|groin", "[LOCATION]", linearized_text, flags=re.IGNORECASE)
            linearized_texts.append(linearized_text)

        inputs = self.tokenizer(
            linearized_texts,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()

        logits_np = logits.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()

        results = []
        for i, text in enumerate(texts):
            pred_id = int(pred_ids[i])
            pred_label = self.id2label[pred_id]
            results.append({
                "text": text,
                "formatted_text": linearized_texts[i],
                "prediction": {
                    "label": pred_label,
                    "id": pred_id,
                    "probability": float(probs_np[i][pred_id]),
                    "probabilities": {self.id2label[j]: float(p) for j, p in enumerate(probs_np[i])},
                    "logits": {self.id2label[j]: float(l) for j, l in enumerate(logits_np[i])},
                }
            })
        return results

    def predict_with_metrics(
        self,
        texts: List[str],
        true_labels: List[str],
        ages: Optional[List[int]] = None,
        genders: Optional[List[str]] = None,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        # Process in batches to avoid CUDA OOM
        all_preds = []
        num_samples = len(texts)
        
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_texts = texts[i:end_idx]
            batch_ages = ages[i:end_idx] if ages else None
            batch_genders = genders[i:end_idx] if genders else None
            
            batch_preds = self.predict_batch(batch_texts, batch_ages, batch_genders)
            all_preds.extend(batch_preds)
        
        pred_labels = [p["prediction"]["label"] for p in all_preds]
        pred_ids = [p["prediction"]["id"] for p in all_preds]
        true_ids = [self.label2id[label] for label in true_labels]

        metrics = self.metric_calculator.compute_metrics((pred_ids, true_ids))
        report = classification_report(
            true_ids,
            pred_ids,
            labels=list(range(len(self.id2label))),
            target_names=list(self.id2label.values()),
            output_dict=True,
            zero_division=0,
        )

        return {
            "predictions": all_preds,
            "metrics": {"basic_metrics": metrics, "classification_report": report},
        }

    def interactive_predict(self):
        print("\n" + "=" * 70)
        mode = "BASELINE (--no_ckpt)" if self.no_ckpt else "FINE-TUNED (checkpoint)"
        print(f" Real-time DistilBERT Prediction | Mode: {mode} | Seed: {self.seed}")
        print("=" * 70)
        print(f"Model: {self.model_config['model']['name']}")
        print(f"Labels: {', '.join(self.id2label.values())}")
        print(f"Max sequence length: {self.max_seq_len}")
        print("=" * 70)
        print("Type 'exit' to quit.")
        print("=" * 70)

        while True:
            try:
                text = input("\nEnter text for prediction: ")
                if text.lower() == "exit":
                    print("\nExiting...")
                    break
                if not text.strip():
                    print("Please enter some text.")
                    continue
                result = self.predict_single(text)
                self._display_prediction_result(result)
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")

    def _display_prediction_result(self, result: Dict[str, Any]):
        print("\n" + "=" * 50)
        print("TEXT")
        print("=" * 50)
        print(result["text"])
        print("\n" + "=" * 50)
        print("PREDICTION RESULT")
        print("=" * 50)
        pred = result["prediction"]
        print(f"Predicted Label: {pred['label']} (ID: {pred['id']})")
        print(f"Confidence: {pred['probability']:.4f}")
        print("\nCLASS PROBABILITIES")
        print("-" * 30)
        sorted_probs = sorted(pred["probabilities"].items(), key=lambda x: x[1], reverse=True)
        for label, prob in sorted_probs:
            print(f"{label}: {prob:.4f}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Real-time prediction with DistilBERT model")
    parser.add_argument("--config", type=str, default="src/config/base_distilbert.yaml",
                        help="Path to configuration file")
    parser.add_argument("--labels_path", type=str, default="preprocess_jsonl/labels.json",
                        help="Path to labels JSON file")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint file or directory (ignored if --no_ckpt)")
    parser.add_argument("--no_ckpt", action="store_true",
                        help="Baseline mode: do not load checkpoint (pretrained encoder + random head)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for reproducibility (important for baseline runs)")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--text", type=str, default=None,
                        help="Single text to predict (non-interactive mode)")
    parser.add_argument("--texts_file", type=str, default=None,
                        help="Path to file with texts for batch prediction")
    parser.add_argument("--labels_file", type=str, default=None,
                        help="Path to file with ground truth labels (for metrics)")

    args = parser.parse_args()

    predictor = RealTimePredictor(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config,
        labels_path=args.labels_path,
        no_ckpt=args.no_ckpt,
        seed=args.seed
    )

    if args.interactive:
        predictor.interactive_predict()
    elif args.text:
        result = predictor.predict_single(args.text)
        predictor._display_prediction_result(result)
    elif args.texts_file:
        with open(args.texts_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        if args.labels_file:
            with open(args.labels_file, "r", encoding="utf-8") as f:
                true_labels = [line.strip() for line in f if line.strip()]

            if len(texts) != len(true_labels):
                print(f"Error: texts ({len(texts)}) != labels ({len(true_labels)})")
                return

            result = predictor.predict_with_metrics(texts, true_labels)
            print("\n" + "=" * 60)
            print("METRICS SUMMARY")
            print("=" * 60)
            print("Basic Metrics:")
            print("-" * 30)
            for metric, value in result["metrics"]["basic_metrics"].items():
                if metric != "loss":
                    print(f"{metric}: {value:.4f}")
        else:
            results = predictor.predict_batch(texts)
            print("\n" + "=" * 60)
            print("BATCH PREDICTION RESULTS")
            print("=" * 60)
            for i, r in enumerate(results):
                t = r["text"]
                print(f"\nPrediction {i+1}:")
                print(f"Text: {t[:100]}..." if len(t) > 100 else f"Text: {t}")
                print(f"Predicted: {r['prediction']['label']} (Confidence: {r['prediction']['probability']:.4f})")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
