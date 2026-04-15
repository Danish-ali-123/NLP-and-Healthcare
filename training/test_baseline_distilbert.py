#!/usr/bin/env python3
"""
Runs DistilBERT baseline (pretrained encoder + random head) for EN/HI/PA for 5 runs each
and saves baseline_results.json with Mean ± Std.

This uses predict_jsonl_distilbert.py RealTimePredictor in --no_ckpt mode.
"""

import json
import numpy as np
from pathlib import Path
import argparse

# Import predictor
from predict_jsonl_distilbert import RealTimePredictor


DEFAULT_PROMPTS = {
    "Arthritis": "joint pain and stiffness, worse in morning",
    "Fracture": "severe pain after fall, swelling and inability to move limb",
    "Osteoporosis": "back pain and history of low bone density, minor trauma pain",
    "Sprain": "twisted ankle, swelling and pain on movement",
    "Tendonitis": "pain near a tendon after repetitive activity",
    "Normal": "no pain, no swelling, normal movement",
}


def compute_stats(values):
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_en", type=str, default="src/config/base_distilbert.yaml")
    parser.add_argument("--config_hi", type=str, default="src/config/base_distilbert.yaml")
    parser.add_argument("--config_pa", type=str, default="src/config/base_distilbert.yaml")
    parser.add_argument("--labels_path", type=str, default="preprocess_jsonl/labels.json")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", type=str, default="train_jsonl/baseline_results.json")
    parser.add_argument("--seed_start", type=int, default=1000,
                        help="Each run uses seed = seed_start + run_idx")
    args = parser.parse_args()

    # If you have separate configs per language, pass them here.
    # If not, same config is fine.
    lang_to_config = {
        "en": args.config_en,
        "hi": args.config_hi,
        "pa": args.config_pa,
    }

    all_results = {}

    for lang in ["en", "hi", "pa"]:
        runs_data = []
        config_path = lang_to_config[lang]

        for r in range(args.runs):
            seed = args.seed_start + r

            predictor = RealTimePredictor(
                checkpoint_path="DUMMY",  # ignored in baseline mode
                config_path=config_path,
                labels_path=args.labels_path,
                no_ckpt=True,
                seed=seed
            )

            run_out = {}
            for k, prompt in DEFAULT_PROMPTS.items():
                pred = predictor.predict_single(prompt)
                run_out[k] = {
                    "predicted_label": pred["prediction"]["label"],
                    "confidence": pred["prediction"]["probability"],
                    "probabilities": pred["prediction"]["probabilities"],
                }
            runs_data.append(run_out)

        # Aggregate stats
        stats = {}
        for k in DEFAULT_PROMPTS.keys():
            confs = [runs_data[i][k]["confidence"] for i in range(args.runs)]
            mean_c, std_c = compute_stats(confs)

            # label distribution
            labels = [runs_data[i][k]["predicted_label"] for i in range(args.runs)]
            label_dist = {}
            for lb in labels:
                label_dist[lb] = label_dist.get(lb, 0) + 1

            # probability stats per class
            # assuming probabilities dict keys are consistent
            class_names = list(runs_data[0][k]["probabilities"].keys())
            prob_stats = {}
            for cname in class_names:
                vals = [runs_data[i][k]["probabilities"][cname] for i in range(args.runs)]
                m, s = compute_stats(vals)
                prob_stats[cname] = {"mean": m, "std": s}

            stats[k] = {
                "mean_confidence": mean_c,
                "std_confidence": std_c,
                "predicted_labels": labels,
                "label_distribution": label_dist,
                "probability_stats": prob_stats
            }

        all_results[lang] = {
            "runs": runs_data,
            "statistics": stats
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved baseline results to: {out_path.resolve()}")
    print("Tip: Std will be non-zero now because we change seeds across runs.")


if __name__ == "__main__":
    main()
