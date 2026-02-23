from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_model_summary_plot(output_dir: Path) -> None:
    metrics = pd.DataFrame(
        [
            {"Model": "Logistic Regression", "Accuracy": 0.8263, "AUC": 0.9050},
            {"Model": "Decision Tree", "Accuracy": 0.9975, "AUC": 0.9989},
            {"Model": "Random Forest", "Accuracy": 0.9983, "AUC": 0.99999},
            {"Model": "KNN", "Accuracy": 0.9070, "AUC": np.nan},
            {"Model": "SVM", "Accuracy": 0.8303, "AUC": np.nan},
            {"Model": "Gradient Boosting", "Accuracy": 0.9950, "AUC": np.nan},
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=200)

    acc = metrics.sort_values("Accuracy", ascending=False)
    axes[0].barh(acc["Model"], acc["Accuracy"], color="#0f766e")
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.75, 1.01)
    axes[0].set_title("Accuracy by Model")
    axes[0].set_xlabel("Accuracy")
    for y, x in enumerate(acc["Accuracy"]):
        axes[0].text(x + 0.002, y, f"{x:.3f}", va="center", fontsize=8)

    auc = metrics.dropna(subset=["AUC"]).sort_values("AUC", ascending=False)
    axes[1].barh(auc["Model"], auc["AUC"], color="#b45309")
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.88, 1.005)
    axes[1].set_title("AUC by Model (Reported)")
    axes[1].set_xlabel("AUC")
    for y, x in enumerate(auc["AUC"]):
        axes[1].text(x + 0.0005, y, f"{x:.4f}", va="center", fontsize=8)

    fig.suptitle("Churn Model Benchmark Summary", fontsize=14, weight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "model_comparison_summary.png", bbox_inches="tight")
    plt.close(fig)


def build_threshold_tradeoff_plot(output_dir: Path) -> None:
    threshold_df = pd.DataFrame(
        [
            {"Threshold": 0.1, "Accuracy": 0.6942, "Precision": 0.6087, "Recall": 0.9745, "F1": 0.7493},
            {"Threshold": 0.3, "Accuracy": 0.7958, "Precision": 0.7241, "Recall": 0.9121, "F1": 0.8073},
            {"Threshold": 0.5, "Accuracy": 0.8267, "Precision": 0.8138, "Recall": 0.8175, "F1": 0.8157},
            {"Threshold": 0.7, "Accuracy": 0.7946, "Precision": 0.8793, "Recall": 0.6515, "F1": 0.7485},
            {"Threshold": 0.9, "Accuracy": 0.6770, "Precision": 0.9576, "Recall": 0.3256, "F1": 0.4860},
        ]
    )

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=200)
    lines = [
        ("Accuracy", "#0f766e"),
        ("Precision", "#1d4ed8"),
        ("Recall", "#dc2626"),
        ("F1", "#a855f7"),
    ]
    for col, color in lines:
        ax.plot(
            threshold_df["Threshold"],
            threshold_df[col],
            marker="o",
            linewidth=2,
            label=col,
            color=color,
        )

    best_row = threshold_df.loc[threshold_df["Accuracy"].idxmax()]
    ax.axvline(best_row["Threshold"], color="#111827", linestyle="--", alpha=0.6)
    ax.text(best_row["Threshold"] + 0.01, 0.61, f"Best accuracy threshold = {best_row['Threshold']:.1f}", fontsize=9)
    ax.set_title("Logistic Regression Threshold Trade-off")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0.3, 1.0)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_dir / "logistic_threshold_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    build_model_summary_plot(output_dir=output_dir)
    build_threshold_tradeoff_plot(output_dir=output_dir)
    print("Generated README visuals in", output_dir)


if __name__ == "__main__":
    main()

