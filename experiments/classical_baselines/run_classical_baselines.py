import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

import sys


EXP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXP_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from project_config import (
    MIN_CONSECUTIVE_WINDOWS,
    NORMAL_HOLDOUT_FILES,
    S,
    T,
    TIME_COL,
    TRUE_INFECTED_END_SEC,
    TRUE_INFECTED_START_SEC,
)


CLEAN_NORMAL_DIR = PROJECT_DIR / "clean_data" / "normal"
CLEAN_INFECTED_DIR = PROJECT_DIR / "clean_data" / "infected"
CLEAN_MIXED_DIR = PROJECT_DIR / "clean_data" / "mixed"
DL_SUMMARY_PATH = PROJECT_DIR / "results" / "tables" / "summary_metrics.csv"

RESULTS_DIR = EXP_DIR / "results"
TABLE_DIR = EXP_DIR / "tables"
FIG_DIR = EXP_DIR / "figures"
MODEL_DIR = EXP_DIR / "models"

RANDOM_STATE = 42


def load_reference_columns() -> list[str]:
    files = sorted(CLEAN_NORMAL_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No normal CSV files in {CLEAN_NORMAL_DIR}")
    df = pd.read_csv(files[0], nrows=1)
    return [c for c in df.columns if c != TIME_COL]


def load_feature_frame(path: Path, reference_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TIME_COL not in df.columns:
        raise ValueError(f"{path.name}: missing {TIME_COL}")

    current_cols = [c for c in df.columns if c != TIME_COL]
    if current_cols != reference_cols:
        missing = sorted(set(reference_cols) - set(current_cols))
        extra = sorted(set(current_cols) - set(reference_cols))
        raise ValueError(
            f"{path.name}: feature columns do not match reference. "
            f"Missing={missing}; extra={extra}"
        )
    return df


def make_window_features(X: np.ndarray) -> np.ndarray:
    n, d = X.shape
    m = (n - T) // S + 1
    if m <= 0:
        return np.zeros((0, d * 5), dtype=np.float32)

    rows = []
    for start in range(0, n - T + 1, S):
        w = X[start:start + T]
        row = np.concatenate(
            [
                w.mean(axis=0),
                w.std(axis=0),
                w.min(axis=0),
                w.max(axis=0),
                w[-1] - w[0],
            ]
        )
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def build_normal_dataset(reference_cols: list[str]) -> dict:
    features = []
    file_summaries = []

    for path in sorted(CLEAN_NORMAL_DIR.glob("*.csv")):
        if path.name in NORMAL_HOLDOUT_FILES:
            continue
        df = load_feature_frame(path, reference_cols)
        X = df[reference_cols].to_numpy(dtype=np.float32)
        F = make_window_features(X)
        if len(F) > 0:
            features.append(F)
        file_summaries.append({"file": path.name, "rows": int(len(df)), "windows": int(len(F))})

    if not features:
        raise ValueError("No normal windows were created")

    all_features = np.concatenate(features, axis=0)
    n_train = int(len(all_features) * 0.8)
    return {
        "X_train": all_features[:n_train],
        "X_val": all_features[n_train:],
        "file_summaries": file_summaries,
        "total_windows": int(len(all_features)),
        "train_windows": int(n_train),
        "val_windows": int(len(all_features) - n_train),
    }


def anomaly_scores(model: Pipeline, X: np.ndarray) -> np.ndarray:
    # sklearn one-class estimators return higher decision_function for normal samples.
    return -model.decision_function(X)


def fit_models(X_train: np.ndarray) -> dict:
    return {
        "isolation_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    IsolationForest(
                        n_estimators=300,
                        contamination="auto",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ).fit(X_train),
        "one_class_svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    OneClassSVM(
                        kernel="rbf",
                        gamma="scale",
                        nu=0.01,
                    ),
                ),
            ]
        ).fit(X_train),
    }


def find_runs(flags: np.ndarray) -> list[tuple[int, int, int]]:
    runs = []
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1, i - start))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1, len(flags) - start))
    return runs


def safe_div(a: int, b: int) -> float:
    return 0.0 if b == 0 else a / b


def true_labels_for_mixed(num_windows: int) -> np.ndarray:
    labels = np.zeros(num_windows, dtype=bool)
    for i in range(num_windows):
        window_start = i * S
        window_end = window_start + T
        labels[i] = (
            window_start < TRUE_INFECTED_END_SEC
            and window_end > TRUE_INFECTED_START_SEC
        )
    return labels


def confusion(predicted: np.ndarray, true_labels: np.ndarray) -> dict:
    tp = int(np.logical_and(predicted, true_labels).sum())
    fp = int(np.logical_and(predicted, ~true_labels).sum())
    fn = int(np.logical_and(~predicted, true_labels).sum())
    tn = int(np.logical_and(~predicted, ~true_labels).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    fpr = safe_div(fp, fp + tn)
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "false_positive_rate": float(fpr),
    }


def summarize_scores(scores: np.ndarray, threshold: float) -> dict:
    predicted = scores > threshold
    events = [run for run in find_runs(predicted) if run[2] >= MIN_CONSECUTIVE_WINDOWS]
    return {
        "windows": int(len(scores)),
        "score_min": float(scores.min()),
        "score_mean": float(scores.mean()),
        "score_max": float(scores.max()),
        "positive_windows": int(predicted.sum()),
        "positive_window_rate": float(predicted.mean()),
        "first_event_start_sec": int(events[0][0] * S) if events else None,
        "events": [
            {
                "start_window": int(start),
                "end_window": int(end),
                "length_windows": int(length),
                "start_sec": int(start * S),
                "end_sec": int(end * S),
            }
            for start, end, length in events
        ],
    }


def evaluate_infected(model_name: str, model: Pipeline, threshold: float, reference_cols: list[str]) -> dict:
    files = []
    for path in sorted(CLEAN_INFECTED_DIR.glob("*.csv")):
        df = load_feature_frame(path, reference_cols)
        X = df[reference_cols].to_numpy(dtype=np.float32)
        F = make_window_features(X)
        scores = anomaly_scores(model, F)
        summary = summarize_scores(scores, threshold)
        summary.update({"file": path.name, "rows": int(len(df))})
        files.append(summary)

    return {
        "model": model_name,
        "threshold": float(threshold),
        "window_config": {"T": T, "stride": S},
        "files": files,
        "aggregate": {
            "mean_positive_rate": float(np.mean([x["positive_window_rate"] for x in files])),
        },
    }


def evaluate_mixed(model_name: str, model: Pipeline, threshold: float, reference_cols: list[str]) -> dict:
    files = []
    for path in sorted(CLEAN_MIXED_DIR.glob("*.csv")):
        df = load_feature_frame(path, reference_cols)
        X = df[reference_cols].to_numpy(dtype=np.float32)
        F = make_window_features(X)
        scores = anomaly_scores(model, F)
        predicted = scores > threshold
        true_labels = true_labels_for_mixed(len(scores))
        cm = confusion(predicted, true_labels)
        summary = summarize_scores(scores, threshold)

        first_event_start_sec = summary["first_event_start_sec"]
        summary.update(
            {
                "file": path.name,
                "rows": int(len(df)),
                "confusion_matrix": {k: cm[k] for k in ["TP", "FP", "FN", "TN"]},
                "metrics": {
                    "precision": cm["precision"],
                    "recall": cm["recall"],
                    "f1_score": cm["f1_score"],
                    "false_positive_rate": cm["false_positive_rate"],
                },
                "detection_delay_sec": (
                    first_event_start_sec - TRUE_INFECTED_START_SEC
                    if first_event_start_sec is not None
                    else None
                ),
            }
        )
        files.append(summary)

    aggregate = {
        "mean_precision": float(np.mean([x["metrics"]["precision"] for x in files])),
        "mean_recall": float(np.mean([x["metrics"]["recall"] for x in files])),
        "mean_f1_score": float(np.mean([x["metrics"]["f1_score"] for x in files])),
        "mean_false_positive_rate": float(np.mean([x["metrics"]["false_positive_rate"] for x in files])),
        "mean_detection_delay_sec": float(np.mean([x["detection_delay_sec"] for x in files])),
    }
    return {
        "model": model_name,
        "threshold": float(threshold),
        "window_config": {"T": T, "stride": S},
        "true_infected_interval_sec": {
            "start": TRUE_INFECTED_START_SEC,
            "end": TRUE_INFECTED_END_SEC,
        },
        "aggregate_metrics": aggregate,
        "files": files,
    }


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary_rows(results: dict) -> list[dict]:
    rows = []
    for model_name, payload in results.items():
        infected = payload["infected"]
        mixed = payload["mixed"]
        agg = mixed["aggregate_metrics"]
        rows.append(
            {
                "model": model_name,
                "T": T,
                "threshold": payload["threshold"],
                "infected_positive_rate": infected["aggregate"]["mean_positive_rate"],
                "mixed_precision": agg["mean_precision"],
                "mixed_recall": agg["mean_recall"],
                "mixed_f1": agg["mean_f1_score"],
                "mixed_FPR": agg["mean_false_positive_rate"],
                "mixed_detection_delay_sec": agg["mean_detection_delay_sec"],
            }
        )
    return rows


def build_dl_comparison(classical_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if DL_SUMMARY_PATH.exists():
        dl_df = pd.read_csv(DL_SUMMARY_PATH)
        mixed = dl_df[dl_df["dataset"] == "mixed"]
        infected = dl_df[dl_df["dataset"] == "infected_only"]
        if not mixed.empty:
            m = mixed.iloc[0]
            infected_rate = float(infected.iloc[0]["mean_positive_rate"]) if not infected.empty else np.nan
            rows.append(
                {
                    "model": "tcn_forecaster_dl",
                    "T": T,
                    "infected_positive_rate": infected_rate,
                    "mixed_precision": float(m["mean_precision"]),
                    "mixed_recall": float(m["mean_recall"]),
                    "mixed_f1": float(m["mean_f1"]),
                    "mixed_FPR": float(m["mean_FPR"]),
                    "mixed_detection_delay_sec": -30.0,
                }
            )

    for _, row in classical_df.iterrows():
        rows.append(
            {
                "model": row["model"],
                "T": row["T"],
                "infected_positive_rate": row["infected_positive_rate"],
                "mixed_precision": row["mixed_precision"],
                "mixed_recall": row["mixed_recall"],
                "mixed_f1": row["mixed_f1"],
                "mixed_FPR": row["mixed_FPR"],
                "mixed_detection_delay_sec": row["mixed_detection_delay_sec"],
            }
        )

    return pd.DataFrame(rows)


def save_tables(summary_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary_csv = TABLE_DIR / "classical_baselines_summary.csv"
    summary_md = TABLE_DIR / "classical_baselines_summary.md"
    comparison_csv = TABLE_DIR / "model_comparison_with_dl.csv"
    comparison_md = TABLE_DIR / "model_comparison_with_dl.md"

    summary_df.to_csv(summary_csv, index=False)
    comparison_df.to_csv(comparison_csv, index=False)
    write_markdown_table(summary_df, summary_md)
    write_markdown_table(comparison_df, comparison_md)

    print("Tables:")
    for path in [summary_csv, summary_md, comparison_csv, comparison_md]:
        print(" -", path)


def save_figures(comparison_df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = comparison_df["model"].tolist()
    x = np.arange(len(labels))

    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.18, comparison_df["mixed_f1"], width=0.36, label="F1")
    plt.bar(x + 0.18, comparison_df["mixed_FPR"], width=0.36, label="FPR")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.title("DL vs classical baselines: F1 and FPR")
    plt.ylabel("metric value")
    plt.legend()
    plt.tight_layout()
    path1 = FIG_DIR / "model_comparison_f1_fpr.png"
    plt.savefig(path1, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(x - 0.18, comparison_df["mixed_precision"], width=0.36, label="precision")
    plt.bar(x + 0.18, comparison_df["mixed_recall"], width=0.36, label="recall")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.title("DL vs classical baselines: precision and recall")
    plt.ylabel("metric value")
    plt.legend()
    plt.tight_layout()
    path2 = FIG_DIR / "model_comparison_precision_recall.png"
    plt.savefig(path2, dpi=300)
    plt.close()

    print("Figures:")
    for path in [path1, path2]:
        print(" -", path)


def save_json_results(results: dict, normal_summary: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "window_config": {"T": T, "stride": S},
        "normal_dataset": {
            "total_windows": normal_summary["total_windows"],
            "train_windows": normal_summary["train_windows"],
            "val_windows": normal_summary["val_windows"],
            "files": normal_summary["file_summaries"],
        },
        "models": results,
    }
    (RESULTS_DIR / "classical_baselines_results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print("Results JSON:", RESULTS_DIR / "classical_baselines_results.json")


def main() -> None:
    for directory in [RESULTS_DIR, TABLE_DIR, FIG_DIR, MODEL_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    reference_cols = load_reference_columns()
    normal_data = build_normal_dataset(reference_cols)

    print("=== Classical baselines ===")
    print(f"T={T}, stride={S}, features/window={normal_data['X_train'].shape[1]}")
    print(
        f"normal windows: total={normal_data['total_windows']} "
        f"train={normal_data['train_windows']} val={normal_data['val_windows']}"
    )
    print()

    models = fit_models(normal_data["X_train"])
    results = {}

    for model_name, model in models.items():
        val_scores = anomaly_scores(model, normal_data["X_val"])
        threshold = float(np.quantile(val_scores, 0.99))
        joblib.dump(model, MODEL_DIR / f"{model_name}.joblib")

        print(f"Model: {model_name}")
        print(f"  validation score mean={val_scores.mean():.6f} max={val_scores.max():.6f}")
        print(f"  threshold p99={threshold:.6f}")

        infected = evaluate_infected(model_name, model, threshold, reference_cols)
        mixed = evaluate_mixed(model_name, model, threshold, reference_cols)
        results[model_name] = {
            "threshold": threshold,
            "infected": infected,
            "mixed": mixed,
        }

        agg = mixed["aggregate_metrics"]
        print(
            f"  mixed precision={agg['mean_precision']:.4f} "
            f"recall={agg['mean_recall']:.4f} "
            f"F1={agg['mean_f1_score']:.4f} "
            f"FPR={agg['mean_false_positive_rate']:.4f} "
            f"delay={agg['mean_detection_delay_sec']:.1f}s"
        )
        print()

    save_json_results(results, normal_data)
    summary_df = pd.DataFrame(build_summary_rows(results))
    comparison_df = build_dl_comparison(summary_df)
    save_tables(summary_df, comparison_df)
    save_figures(comparison_df)


if __name__ == "__main__":
    main()
