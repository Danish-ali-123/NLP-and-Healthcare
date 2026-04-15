import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
CONFIG = "src/config/base_mdeberta.yaml"

def main() -> None:
    cfg_path = PROJECT_ROOT / CONFIG
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    print("\n====================================================")
    print(f"🚀 TRAINING mDeBERTa with config: {cfg_path}")
    print("====================================================\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m", "src.train.train",
            "--config", str(cfg_path),
        ],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"mDeBERTa training failed with code {result.returncode}"
        )

if __name__ == "__main__":
    main()