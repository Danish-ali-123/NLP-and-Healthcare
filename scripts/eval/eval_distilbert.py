import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

def eval_single_model(model_root_rel: str, output_rel: str, label: str) -> None:
    """Helper to evaluate a single model."""
    model_root = PROJECT_ROOT / model_root_rel
    output_dir = PROJECT_ROOT / output_rel
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n====================================================")
    print(f"🧪 Evaluating model: {label}")
    print(f"Model root:   {model_root}")
    print(f"Output dir:   {output_dir}")
    print("====================================================\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m", "src.eval.run_single_supervised",
            "--model_root", str(model_root),
            "--output_dir", str(output_dir),
        ],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"Evaluation failed for {label} with code {result.returncode}"
        )

def find_latest_experiment(backbone_dir: Path) -> str:
    """Find the most recent experiment directory."""
    if not backbone_dir.exists():
        raise FileNotFoundError(f"Backbone directory not found: {backbone_dir}")
    
    experiment_dirs = [
        d for d in backbone_dir.iterdir()
        if d.is_dir() and d.name.startswith('E')
    ]
    
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiments found in {backbone_dir}")
    
    # Sort by creation time (most recent first)
    experiment_dirs_with_time = [
        (d, os.path.getctime(d))
        for d in experiment_dirs
    ]
    experiment_dirs_with_time.sort(key=lambda x: x[1], reverse=True)
    
    return experiment_dirs_with_time[0][0].name

def main() -> None:
    backbone_dir = PROJECT_ROOT / "experiments" / "distilbert"
    latest_run = find_latest_experiment(backbone_dir)
    
    print(f"📊 Found latest DistilBERT experiment: {latest_run}")
    
    for lang in ['en', 'hi', 'pa']:
        model_root_rel = f"experiments/distilbert/{latest_run}/{lang}"
        output_rel = f"reports/eval_supervised/distilbert/{latest_run}/{lang}"
        label = f"DistilBERT / {latest_run} / {lang}"
        
        eval_single_model(model_root_rel, output_rel, label)

if __name__ == "__main__":
    main()

