import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Configs for all backbones
CONFIGS = [
    "src/config/base_distilbert.yaml",
    "src/config/base_indicbert.yaml",
    "src/config/base_indicbert_hpa.yaml",
    "src/config/base_xlmroberta.yaml",
    "src/config/base_mdeberta.yaml",
]

def run_config(config_rel: str) -> None:
    cfg_path = PROJECT_ROOT / config_rel
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    print("\n====================================================")
    print(f"🚀 TRAINING with config: {config_rel}")
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
            f"Training failed for {cfg_path} with code {result.returncode}"
        )

def main() -> None:
    print(f"📦 Project root: {PROJECT_ROOT}")
    for cfg in CONFIGS:
        run_config(cfg)

if __name__ == "__main__":
    main()

