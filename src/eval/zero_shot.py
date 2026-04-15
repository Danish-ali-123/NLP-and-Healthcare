import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import MultilingualClinicalDataset, DataCollator, load_label_mapping
from src.models.baselines import build_model_and_tokenizer
from src.models.metrics import MetricCalculator
from src.evaluate.evaluate import load_model_from_checkpoint


logger = logging.getLogger(__name__)


MODEL_CONFIG_PATHS: Dict[str, str] = {
    "distilbert": "src/config/base_distilbert.yaml",
    "indicbert": "src/config/base_indicbert_hpa.yaml",
    "xlm_roberta": "src/config/base_xlmroberta.yaml",
    "mdeberta": "src/config/base_mdeberta.yaml",
}


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_zero_shot_model_and_tokenizer(
    base_config: Dict[str, Any],
    model_type: str,
    label_info: Dict[str, Any],
    device: torch.device,
    use_trained_checkpoint: bool,
    run_id: str,
) -> Tuple[torch.nn.Module, Any]:
    """
    Build model + tokenizer either from a trained checkpoint or from the
    raw pretrained backbone for true zero-shot evaluation.
    """
    if model_type not in MODEL_CONFIG_PATHS:
        raise ValueError(f"Unsupported model_type '{model_type}'. "
                         f"Expected one of {list(MODEL_CONFIG_PATHS.keys())}")

    model_config = load_yaml(MODEL_CONFIG_PATHS[model_type])

    # Merge with base config (reuse logic from train.train.merge_configs)
    merged_config: Dict[str, Any] = base_config.copy()
    if "training" in model_config:
        if "training" not in merged_config:
            merged_config["training"] = {}
        merged_config["training"].update(model_config["training"])
    if "model" in model_config:
        merged_config["model"] = model_config["model"]

    if use_trained_checkpoint:
        # Trained checkpoint layout:
        #   experiments/<model_type>/<run_id>/<language>/best.ckpt
        # The concrete language-specific path is resolved by the caller.
        # Here we just return a model builder that can be reused per-language.
        def _build_from_checkpoint(lang: str) -> Tuple[torch.nn.Module, Any]:
            model_root = os.path.join(base_config["paths"]["experiments"], model_type, run_id, lang)
            checkpoint_path = os.path.join(model_root, "best.ckpt")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint not found for {model_type} (run_id={run_id}, lang={lang}): {checkpoint_path}"
                )
            logger.info(f"Loading trained checkpoint from {checkpoint_path}")
            model, tokenizer = load_model_from_checkpoint(
                checkpoint_path=checkpoint_path,
                model_config=merged_config,
                label_info=label_info,
                device=device,
            )
            return model, tokenizer

        return _build_from_checkpoint, None  # type: ignore[return-value]

    # True zero-shot: build from HF backbone only
    logger.info(f"Building zero-shot model for backbone '{model_type}'")
    model, tokenizer = build_model_and_tokenizer(
        model_config=merged_config,
        label_info=label_info,
        device=device,
    )
    return model, tokenizer


def run_inference_zero_shot(
    model: torch.nn.Module,
    tokenizer: Any,
    dataset: MultilingualClinicalDataset,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and return (y_true, y_pred)."""
    data_collator = DataCollator(tokenizer, padding="longest")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=data_collator,
    )

    all_true: List[int] = []
    all_pred: List[int] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Zero-shot inference"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())

    return np.array(all_true), np.array(all_pred)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot or checkpoint-based evaluation on test.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/base.yaml",
        help="Path to base configuration file",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["distilbert", "indicbert", "xlm_roberta", "mdeberta"],
        help="Backbone to evaluate",
    )
    parser.add_argument(
        "--use_trained_checkpoint",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=False,
        help="If true, load best.ckpt from experiments/<model_type>/<run_id>/<lang>/",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run id (E######## style) when using trained checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu; default: auto)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Configs and labels
    base_config = load_yaml(args.config)
    processed_dir = base_config["paths"]["data_processed"]
    labels_path = os.path.join(processed_dir, "labels.json")
    label_info = load_label_mapping(labels_path)

    # Languages
    languages: List[str] = base_config.get("data", {}).get("languages", ["en", "hi", "pa"])

    # Build model/tokenizer (or checkpoint builder)
    if args.use_trained_checkpoint:
        if not args.run_id:
            raise ValueError("--run_id is required when --use_trained_checkpoint=true")
        model_builder, _ = build_zero_shot_model_and_tokenizer(
            base_config=base_config,
            model_type=args.model_type,
            label_info=label_info,
            device=device,
            use_trained_checkpoint=True,
            run_id=args.run_id,
        )
    else:
        model, tokenizer = build_zero_shot_model_and_tokenizer(
            base_config=base_config,
            model_type=args.model_type,
            label_info=label_info,
            device=device,
            use_trained_checkpoint=False,
            run_id="",
        )

    metric_calculator = MetricCalculator(label_info["id2label"])

    all_language_results: Dict[str, Dict[str, float]] = {}

    test_path = os.path.join(processed_dir, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")

    for lang in languages:
        logger.info("=" * 60)
        logger.info(f"Evaluating language: {lang.upper()}")
        logger.info("=" * 60)

        if args.use_trained_checkpoint:
            model, tokenizer = model_builder(lang)  # type: ignore[operator]

        max_seq_len = base_config.get("data", {}).get("max_seq_len", 256)
        if "model" in base_config and "tokenizer" in base_config["model"]:
            max_seq_len = base_config["model"]["tokenizer"].get("max_length", max_seq_len)

        dataset = MultilingualClinicalDataset(
            test_path,
            tokenizer,
            label_info["label2id"],
            max_length=max_seq_len,
            language=lang,
        )

        if len(dataset) == 0:
            logger.warning(f"No test examples found for language '{lang}', skipping.")
            continue

        y_true, y_pred = run_inference_zero_shot(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            device=device,
            batch_size=int(base_config.get("training", {}).get("batch_size_effective", 32)),
        )

        metrics = metric_calculator.compute_metrics((y_pred, y_true))
        all_language_results[lang] = metrics

        logger.info(f"Metrics for {lang}: {metrics}")

    # Save summary under experiments/<model_type>/<run_id or 'zero_shot'>/
    run_id = args.run_id if args.use_trained_checkpoint and args.run_id else "zero_shot"
    exp_root = os.path.join(base_config["paths"]["experiments"], args.model_type, run_id)
    os.makedirs(exp_root, exist_ok=True)

    summary_path = Path(exp_root) / "zero_shot_metrics.json"
    summary = {
        "model_type": args.model_type,
        "run_id": run_id,
        "use_trained_checkpoint": args.use_trained_checkpoint,
        "languages": list(all_language_results.keys()),
        "metrics": all_language_results,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        import json

        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved zero-shot metrics to {summary_path}")


if __name__ == "__main__":
    main()


