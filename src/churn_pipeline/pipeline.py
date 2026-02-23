from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set_theme(style="whitegrid")

TARGET_COLUMN = "Churn"
LEAKAGE_COLUMNS = [
    "Churn Category",
    "Churn Reason",
    "Customer Status",
    "Churn Score",
]
NON_FEATURE_COLUMNS = [
    "Customer ID",
    "City",
    "State",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Country",
    "Zip Code",
    "Quarter",
]


@dataclass
class DatasetBundle:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_columns: list[str]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load source CSV and normalize simple data typing issues."""
    df = pd.read_csv(path)

    # Force bool-like columns to integer where relevant.
    bool_like_columns = [
        "Referred a Friend",
        "Phone Service",
        "Internet Service",
        "Paperless Billing",
        "Online Security",
        "Online Backup",
        "Device Protection Plan",
        "Premium Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Streaming Music",
        "Unlimited Data",
        "Under 30",
        "Married",
        "Dependents",
        "Partner",
        "Multiple Lines",
        "Senior Citizen",
    ]
    for col in bool_like_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in dataset.")

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    y = df[TARGET_COLUMN].copy()

    drop_cols = [TARGET_COLUMN]
    drop_cols.extend([c for c in LEAKAGE_COLUMNS if c in df.columns])
    drop_cols.extend([c for c in NON_FEATURE_COLUMNS if c in df.columns])

    X = df.drop(columns=drop_cols).copy()

    # Drop columns that are mostly missing.
    min_non_null_ratio = 0.6
    non_null_ratio = X.notna().mean()
    keep_cols = non_null_ratio[non_null_ratio >= min_non_null_ratio].index.tolist()
    X = X[keep_cols]

    # Remove constant columns.
    nunique = X.nunique(dropna=True)
    X = X.loc[:, nunique > 1]

    return X, y, X.columns.tolist()


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def build_models(preprocessor: ColumnTransformer, random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def run_model_comparison(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any], DatasetBundle]:
    """Train multiple models, return score table and fitted artifacts."""
    X, y, feature_columns = _prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = _build_preprocessor(X_train)
    models = build_models(preprocessor=preprocessor, random_state=random_state)

    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    predictions: dict[str, dict[str, np.ndarray]] = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = _compute_metrics(y_test, y_pred, y_prob)
        rows.append({"model": model_name, **metrics})
        fitted_models[model_name] = model
        predictions[model_name] = {"y_pred": y_pred, "y_prob": y_prob}

    scores = (
        pd.DataFrame(rows)
        .sort_values(by=["f1", "recall", "roc_auc"], ascending=False)
        .reset_index(drop=True)
    )

    best_model_name = str(scores.iloc[0]["model"])
    bundle = DatasetBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_columns=feature_columns,
    )
    artifacts: dict[str, Any] = {
        "models": fitted_models,
        "predictions": predictions,
        "best_model_name": best_model_name,
        "best_model": fitted_models[best_model_name],
    }
    return scores, artifacts, bundle


def create_eda_figures(df: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "Contract" in df.columns:
        contract_rates = (
            df.groupby("Contract", observed=False)[TARGET_COLUMN]
            .mean()
            .sort_values(ascending=False)
            .mul(100)
        )
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(
            x=contract_rates.index,
            y=contract_rates.values,
            ax=ax,
            palette="crest",
            hue=contract_rates.index,
            legend=False,
        )
        ax.set_title("Churn Rate by Contract Type")
        ax.set_xlabel("Contract")
        ax.set_ylabel("Churn Rate (%)")
        plt.xticks(rotation=20)
        plt.tight_layout()
        fig.savefig(output_dir / "churn_rate_by_contract.png", dpi=220)
        plt.close(fig)

    if "Tenure in Months" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.kdeplot(
            data=df,
            x="Tenure in Months",
            hue=TARGET_COLUMN,
            fill=True,
            common_norm=False,
            alpha=0.35,
            ax=ax,
        )
        ax.set_title("Tenure Distribution by Churn Outcome")
        ax.set_xlabel("Tenure (Months)")
        ax.set_ylabel("Density")
        ax.legend(title="Churn", labels=["Stayed (0)", "Churned (1)"])
        plt.tight_layout()
        fig.savefig(output_dir / "tenure_distribution_by_churn.png", dpi=220)
        plt.close(fig)

    geo_cols = ["Latitude", "Longitude", "City"]
    if all(col in df.columns for col in geo_cols):
        city_geo = (
            df.groupby("City", observed=False)
            .agg(
                churn_rate=(TARGET_COLUMN, "mean"),
                latitude=("Latitude", "mean"),
                longitude=("Longitude", "mean"),
                customers=(TARGET_COLUMN, "size"),
            )
            .reset_index()
        )
        city_geo = city_geo[city_geo["customers"] >= 8].copy()
        city_geo["churn_rate_pct"] = city_geo["churn_rate"] * 100

        fig, ax = plt.subplots(figsize=(9.5, 6))
        scatter = ax.scatter(
            city_geo["longitude"],
            city_geo["latitude"],
            s=np.clip(city_geo["customers"] * 0.8, 20, 300),
            c=city_geo["churn_rate_pct"],
            cmap="magma",
            alpha=0.8,
            edgecolor="black",
            linewidth=0.25,
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Churn Rate (%)")
        ax.set_title("California Churn Hotspots (City-level)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.2)
        plt.tight_layout()
        fig.savefig(output_dir / "california_churn_hotspots.png", dpi=220)
        plt.close(fig)


def save_model_diagnostics(
    model: Pipeline,
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"{model_name}: Confusion Matrix")
    plt.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_best_model.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, color="#0f766e")
    ax.set_title(f"{model_name}: ROC Curve")
    plt.tight_layout()
    fig.savefig(output_dir / "roc_curve_best_model.png", dpi=220)
    plt.close(fig)

    model_step = model.named_steps.get("model")
    preprocessor = model.named_steps.get("preprocessor")
    if preprocessor is None:
        return

    if isinstance(model_step, RandomForestClassifier):
        importances = model_step.feature_importances_
        feature_names = preprocessor.get_feature_names_out()
        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values(by="importance", ascending=False)
            .head(15)
        )

        fig, ax = plt.subplots(figsize=(8.5, 6))
        sns.barplot(
            data=importance_df,
            x="importance",
            y="feature",
            hue="feature",
            palette="viridis",
            legend=False,
            ax=ax,
        )
        ax.set_title("Top Feature Importance (Random Forest)")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        fig.savefig(output_dir / "top_feature_importance.png", dpi=220)
        plt.close(fig)
        return

    if isinstance(model_step, LogisticRegression):
        feature_names = preprocessor.get_feature_names_out()
        coefs = model_step.coef_.ravel()
        coef_df = (
            pd.DataFrame({"feature": feature_names, "coefficient": coefs})
            .assign(abs_coef=lambda x: x["coefficient"].abs())
            .sort_values("abs_coef", ascending=False)
            .head(15)
        )
        coef_df["direction"] = np.where(coef_df["coefficient"] >= 0, "increases", "decreases")

        fig, ax = plt.subplots(figsize=(8.5, 6))
        sns.barplot(
            data=coef_df,
            x="coefficient",
            y="feature",
            hue="direction",
            palette={"increases": "#b91c1c", "decreases": "#0f766e"},
            ax=ax,
        )
        ax.set_title("Top Churn Drivers (Logistic Coefficients)")
        ax.set_xlabel("Coefficient")
        ax.set_ylabel("Feature")
        ax.legend(title="Effect on churn odds")
        plt.tight_layout()
        fig.savefig(output_dir / "top_feature_importance.png", dpi=220)
        plt.close(fig)
