import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
CONFIG = "src/config/base_distilbert.yaml"


def main() -> None:
    """Run DistilBERT training on JSONL data."""
    cfg_path = PROJECT_ROOT / CONFIG
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    print("\n" + "="*60)
    print(f"🚀 TRAINING DistilBERT on JSONL data")
    print(f"Config: {cfg_path}")
    print("="*60 + "\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m", "train_jsonl.train_jsonl_distilbert",
            "--config", str(cfg_path),
            "--jsonl_path", "preprocess_jsonl/english_data.jsonl",
            "--labels_path", "preprocess_jsonl/labels.json",
            "--output_dir", "train_jsonl/experiments",
            "--language", "en",
        ],
        cwd=PROJECT_ROOT,
    )
    
    if result.returncode != 0:
        raise SystemExit(
            f"DistilBERT training on JSONL data failed with code {result.returncode}"
        )
    
    print("\n" + "="*60)
    print(f"✅ DistilBERT training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
