"""Generate a model card through the real end-to-end workflow for CI artifacts."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datapilot.config import Settings
from datapilot.data import load_sample
from datapilot.workflow import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("build/example-model-card.md"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="datapilot-model-card-") as temporary:
        root = Path(temporary)
        settings = Settings(
            artifact_root=root / "artifacts",
            database_url=f"sqlite:///{(root / 'runs.db').as_posix()}",
            optuna_trials=0,
            max_critic_retries=0,
        )
        frame, target, name = load_sample("iris")
        result = run_analysis(frame, target, name, settings)
        shutil.copyfile(Path(result.artifacts["model_card"]), args.output)

    print(f"Generated {args.output}")


if __name__ == "__main__":
    main()
