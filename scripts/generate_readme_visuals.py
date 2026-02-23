from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


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


def build_retention_impact_plot(output_dir: Path) -> None:
    # Confusion matrix values from the notebook at threshold ~= 0.5
    # [[TN, FP], [FN, TP]] = [[5708, 1129], [1102, 4936]]
    values = {
        "True Positives (churn correctly flagged)": 4936,
        "False Positives (unnecessary outreach)": 1129,
        "False Negatives (missed churners)": 1102,
        "True Negatives (correctly ignored)": 5708,
    }

    labels = list(values.keys())
    counts = list(values.values())
    colors = ["#0f766e", "#b45309", "#b91c1c", "#334155"]

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
    bars = ax.barh(labels, counts, color=colors)
    ax.invert_yaxis()
    ax.set_title("Operational Impact at Chosen Threshold (0.5)")
    ax.set_xlabel("Number of Customers")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 60, bar.get_y() + bar.get_height() / 2, f"{count:,}", va="center", fontsize=9)

    total_churners = values["True Positives (churn correctly flagged)"] + values["False Negatives (missed churners)"]
    caught_pct = values["True Positives (churn correctly flagged)"] / total_churners
    ax.text(
        0.01,
        -0.22,
        f"Interpretation: the model catches {caught_pct:.1%} of churners while generating {values['False Positives (unnecessary outreach)']:,} false alerts.",
        transform=ax.transAxes,
        fontsize=9,
    )
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    fig.savefig(output_dir / "retention_impact_threshold_05.png", bbox_inches="tight")
    plt.close(fig)


def _draw_step(ax, x: float, y: float, w: float, h: float, title: str, subtitle: str, color: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.6,
        edgecolor="#111827",
        facecolor=color,
        alpha=0.95,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", fontsize=10.5, weight="bold", color="#111827")
    ax.text(x + w / 2, y + h * 0.32, subtitle, ha="center", va="center", fontsize=8.8, color="#1f2937")


def build_workflow_diagram(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 6), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Phase bands
    bands = [
        (0.04, 0.72, 0.92, 0.18, "#dbeafe", "Phase 1: Data and Feature Preparation"),
        (0.04, 0.44, 0.92, 0.18, "#dcfce7", "Phase 2: Modeling and Evaluation"),
        (0.04, 0.16, 0.92, 0.18, "#fef3c7", "Phase 3: Decision and Business Action"),
    ]
    for x, y, w, h, color, label in bands:
        band = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", linewidth=0, facecolor=color, alpha=0.55)
        ax.add_patch(band)
        ax.text(x + 0.01, y + h - 0.035, label, ha="left", va="center", fontsize=10, weight="bold", color="#0f172a")

    # Steps
    box_w, box_h = 0.21, 0.11
    steps = [
        (0.08, 0.75, "1) Raw Data Input", "Customer behavior + plan info", "#bfdbfe"),
        (0.39, 0.75, "2) Cleaning", "Null checks, duplicates, encoding", "#bfdbfe"),
        (0.70, 0.75, "3) Feature Matrix", "Model-ready X and y", "#bfdbfe"),
        (0.08, 0.47, "4) Baseline Model", "Logistic Regression", "#bbf7d0"),
        (0.39, 0.47, "5) Model Benchmark", "Tree, RF, KNN, SVM, GBM", "#bbf7d0"),
        (0.70, 0.47, "6) Threshold Tuning", "Precision vs recall trade-off", "#bbf7d0"),
        (0.23, 0.19, "7) Impact Estimation", "TP / FP / FN operational view", "#fde68a"),
        (0.55, 0.19, "8) Retention Plan", "Prioritize outreach by churn risk", "#fde68a"),
    ]
    for x, y, title, subtitle, color in steps:
        _draw_step(ax, x, y, box_w, box_h, title, subtitle, color)

    # Arrows
    arrows = [
        ((0.29, 0.805), (0.39, 0.805)),
        ((0.60, 0.805), (0.70, 0.805)),
        ((0.185, 0.75), (0.185, 0.58)),
        ((0.495, 0.75), (0.495, 0.58)),
        ((0.805, 0.75), (0.805, 0.58)),
        ((0.29, 0.525), (0.39, 0.525)),
        ((0.60, 0.525), (0.70, 0.525)),
        ((0.805, 0.47), (0.66, 0.30)),
        ((0.495, 0.47), (0.34, 0.30)),
        ((0.44, 0.245), (0.55, 0.245)),
    ]
    for start, end in arrows:
        arrow = FancyArrowPatch(
            posA=start,
            posB=end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.4,
            color="#1f2937",
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arrow)

    ax.text(
        0.5,
        0.06,
        "Notebook flow: from data preparation to actionable retention decisions",
        ha="center",
        va="center",
        fontsize=10,
        color="#334155",
    )
    ax.set_title("Churn Pipeline Workflow", fontsize=16, weight="bold", pad=16)
    fig.savefig(output_dir / "workflow_pipeline_visual.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    build_model_summary_plot(output_dir=output_dir)
    build_threshold_tradeoff_plot(output_dir=output_dir)
    build_retention_impact_plot(output_dir=output_dir)
    build_workflow_diagram(output_dir=output_dir)
    print("Generated README visuals in", output_dir)


if __name__ == "__main__":
    main()
