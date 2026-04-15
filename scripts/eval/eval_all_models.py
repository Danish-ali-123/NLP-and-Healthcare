import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

def main() -> None:
    output_dir = PROJECT_ROOT / "reports" / "eval_supervised"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n====================================================")
    print("🧪 Running supervised evaluation on TEST set for ALL models...")
    print("====================================================\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m", "src.eval.run_all_supervised",
            "--experiments_root", "experiments",
            "--output_dir", str(output_dir.relative_to(PROJECT_ROOT)),
        ],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"Supervised evaluation failed with code {result.returncode}"
        )

if __name__ == "__main__":
    main()

