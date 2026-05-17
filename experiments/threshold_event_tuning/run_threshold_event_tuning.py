import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


EXP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXP_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from project_config import H, S, SCORE_KEY, T, TRUE_INFECTED_END_SEC, TRUE_INFECTED_START_SEC


VAL_SCORES_PATH = PROJECT_DIR / "checkpoints" / "val_scores_from_script.pt"
MIXED_RESULTS_PATH = PROJECT_DIR / "results" / "mixed_dataset_evaluation.json"

TABLE_DIR = EXP_DIR / "tables"
FIG_DIR = EXP_DIR / "figures"

QUANTILES = [0.99, 0.9925, 0.995, 0.997, 0.999]
MIN_CONSECUTIVE_VALUES = [1, 2, 3, 4, 5, 6, 8, 10]


def to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy()
    return np.asarray(value)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def scores_path(relative_path: str) -> Path:
    return PROJECT_DIR / Path(relative_path)


def safe_div(a: int, b: int) -> float:
    return 0.0 if b == 0 else a / b


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


def expand_event_runs(num_windows: int, runs: list[tuple[int, int, int]], min_consecutive: int) -> np.ndarray:
    flags = np.zeros(num_windows, dtype=bool)
    for start, end, length in runs:
        if length >= min_consecutive:
            flags[start:end + 1] = True
    return flags


def true_labels(num_windows: int) -> np.ndarray:
    labels = np.zeros(num_windows, dtype=bool)
    for i in range(num_windows):
        start = i * S
        end = start + T + H
        labels[i] = start < TRUE_INFECTED_END_SEC and end > TRUE_INFECTED_START_SEC
    return labels


def metrics(predicted: np.ndarray, labels: np.ndarray) -> dict:
    tp = int(np.logical_and(predicted, labels).sum())
    fp = int(np.logical_and(predicted, ~labels).sum())
    fn = int(np.logical_and(~predicted, labels).sum())
    tn = int(np.logical_and(~predicted, ~labels).sum())
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


def evaluate_config(mixed_results: dict, threshold: float, min_consecutive: int) -> dict:
    per_file = []
    for item in mixed_results["files"]:
        data = torch.load(scores_path(item["scores_path"]), map_location="cpu", weights_only=False)
        score = to_numpy(data[SCORE_KEY])
        raw_positive = score > threshold
        runs = find_runs(raw_positive)
        event_positive = expand_event_runs(len(score), runs, min_consecutive)
        labels = true_labels(len(score))
        m = metrics(event_positive, labels)
        valid_events = [run for run in runs if run[2] >= min_consecutive]
        first_event = int(valid_events[0][0] * S) if valid_events else None
        delay = first_event - TRUE_INFECTED_START_SEC if first_event is not None else None
        per_file.append(
            {
                **m,
                "first_event_start_sec": first_event,
                "detection_delay_sec": delay,
                "events": len(valid_events),
            }
        )

    return {
        "mean_precision": float(np.mean([x["precision"] for x in per_file])),
        "mean_recall": float(np.mean([x["recall"] for x in per_file])),
        "mean_f1": float(np.mean([x["f1_score"] for x in per_file])),
        "mean_FPR": float(np.mean([x["false_positive_rate"] for x in per_file])),
        "mean_detection_delay_sec": float(np.mean([x["detection_delay_sec"] for x in per_file if x["detection_delay_sec"] is not None])),
        "mean_events": float(np.mean([x["events"] for x in per_file])),
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


def save_figures(df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for min_consecutive, sub in df.groupby("min_consecutive"):
        plt.plot(sub["quantile"], sub["mean_f1"], marker="o", label=f"F1 min_run={min_consecutive}")
    plt.xlabel("validation threshold quantile")
    plt.ylabel("mean F1")
    plt.ylim(0, 1.05)
    plt.title("Threshold/event tuning: F1")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "threshold_event_tuning_f1.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    for min_consecutive, sub in df.groupby("min_consecutive"):
        plt.plot(sub["quantile"], sub["mean_FPR"], marker="o", label=f"FPR min_run={min_consecutive}")
    plt.xlabel("validation threshold quantile")
    plt.ylabel("mean FPR")
    plt.title("Threshold/event tuning: FPR")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "threshold_event_tuning_fpr.png", dpi=300)
    plt.close()


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    mixed_results = load_json(MIXED_RESULTS_PATH)
    val_data = torch.load(VAL_SCORES_PATH, map_location="cpu", weights_only=False)
    val_score = to_numpy(val_data["std_score"])

    rows = []
    for quantile in QUANTILES:
        threshold = float(np.quantile(val_score, quantile))
        for min_consecutive in MIN_CONSECUTIVE_VALUES:
            row = {
                "quantile": quantile,
                "threshold": threshold,
                "min_consecutive": min_consecutive,
                **evaluate_config(mixed_results, threshold, min_consecutive),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df["eligible"] = (df["mean_recall"] >= 0.98).astype(int)
    df = df.sort_values(
        ["eligible", "mean_f1", "mean_FPR", "mean_detection_delay_sec"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    csv_path = TABLE_DIR / "threshold_event_tuning.csv"
    md_path = TABLE_DIR / "threshold_event_tuning.md"
    df.to_csv(csv_path, index=False)
    write_markdown_table(df.head(20), md_path)
    save_figures(df.sort_values(["min_consecutive", "quantile"]))

    print("Top tuning candidates:")
    print(
        df.head(12)[
            [
                "quantile",
                "threshold",
                "min_consecutive",
                "mean_precision",
                "mean_recall",
                "mean_f1",
                "mean_FPR",
                "mean_detection_delay_sec",
                "mean_events",
            ]
        ].to_string(index=False)
    )
    print()
    print("Saved:")
    print(" -", csv_path)
    print(" -", md_path)
    print(" -", FIG_DIR / "threshold_event_tuning_f1.png")
    print(" -", FIG_DIR / "threshold_event_tuning_fpr.png")


if __name__ == "__main__":
    main()
