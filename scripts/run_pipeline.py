import argparse
import json
from pathlib import Path
import sys

import joblib

# Allow running from repository root without installing as a package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from churn_pipeline.pipeline import (
    create_eda_figures,
    load_data,
    run_model_comparison,
    save_model_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run churn modeling pipeline.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/telco_churn.csv",
        help="Path to source CSV.",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory for metrics and visual outputs.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory for serialized model artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    figures_dir = reports_dir / "figures"
    models_dir = Path(args.models_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data_path)
    create_eda_figures(df=df, output_dir=figures_dir)

    scores, artifacts, bundle = run_model_comparison(df=df)
    scores.to_csv(reports_dir / "model_metrics.csv", index=False)

    metrics_json = scores.to_dict(orient="records")
    with (reports_dir / "model_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics_json, fp, indent=2)

    best_name = artifacts["best_model_name"]
    best_model = artifacts["best_model"]
    preds = artifacts["predictions"][best_name]
    save_model_diagnostics(
        model=best_model,
        model_name=best_name,
        y_true=bundle.y_test,
        y_pred=preds["y_pred"],
        y_prob=preds["y_prob"],
        output_dir=figures_dir,
    )

    joblib.dump(best_model, models_dir / "best_model.joblib")
    print("Pipeline run complete.")
    print(f"Best model: {best_name}")
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
