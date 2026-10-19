import argparse
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-open a completed scLAMP-SR run and export exact encoder means."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--model-script",
        default=(
            "models/"
            "scAPI_shared_feature_lambda_continuous_prior_gentle_zi_"
            "state_contrast_residual.py"
        ),
    )
    parser.add_argument("--checkpoint-path")
    return parser.parse_args()


def first_existing(*paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of the expected files exist: " + ", ".join(map(str, paths)))


def load_model_module(path):
    path = path.resolve()
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(Path.cwd()))
    spec = importlib.util.spec_from_file_location("sclamp_sr_export_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    config_path = first_existing(run_dir / "运行配置.json", run_dir / "run_config.json")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    seed = int(config["seed"])
    checkpoint = (
        Path(args.checkpoint_path).resolve()
        if args.checkpoint_path
        else first_existing(
            run_dir / f"模型_随机种子_{seed}.pt",
            run_dir / f"feature_lambda_model_seed_{seed}.pt",
        )
    )
    config.update(
        {
            "analyze_only": True,
            "model_path": str(checkpoint),
            "output_dir": str(run_dir.parent),
            "run_dir_name": run_dir.name,
        }
    )
    model_module = load_model_module(Path(args.model_script))
    model_module.main(Namespace(**config))


if __name__ == "__main__":
    main()

