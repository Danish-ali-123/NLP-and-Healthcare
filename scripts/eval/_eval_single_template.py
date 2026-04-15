import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

def eval_single_model(model_root_rel: str, output_rel: str, label: str) -> None:
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

