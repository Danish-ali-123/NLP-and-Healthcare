import logging
from typing import Dict, Any, List

# NOTE:
# This module is kept for backwards compatibility and high-level experiment
# orchestration only. The primary training entrypoint is `src/train/train.py`.
# It is NOT wired to the new per-backbone experiment layout and should not be
# used for new runs without further refactoring.

logger = logging.getLogger(__name__)

def run_all_experiments() -> List[Dict[str, Any]]:
    """
    Legacy stub for running multiple experiments.

    The original implementation depended on an older training API (setup_training)
    and a flat experiments/E######## layout. That code path has been retired.

    For new experiments, prefer calling `python -m src.train.train` directly with
    the appropriate configuration.
    """
    logger.warning(
        "run_all_experiments is a legacy helper and is not wired to the current "
        "training pipeline. Use `python -m src.train.train` for new runs."
    )
    return []

def main():
    """Main function to run all experiments."""
    logger.info("Starting all experiments...")
    run_all_experiments()
    logger.info("All experiments completed!")

if __name__ == "__main__":
    main()