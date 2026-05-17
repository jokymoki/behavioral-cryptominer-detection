import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
TABLE_DIR = RESULTS_DIR / "tables"
FIG_DIR = BASE_DIR / "figures" / "report"

VAL_SCORES_PATH = BASE_DIR / "checkpoints" / "val_scores_from_script.pt"
STATS_PATH = BASE_DIR / "checkpoints" / "score_stats.json"
INFECTED_RESULTS_PATH = RESULTS_DIR / "infected_dataset_evaluation.json"
MIXED_RESULTS_PATH = RESULTS_DIR / "mixed_dataset_evaluation.json"

VAL_SCORE_KEY = "std_score"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy()
    return np.asarray(value)


def scores_path(relative_path: str) -> Path:
    return BASE_DIR / Path(relative_path)


def short_name(file_name: str) -> str:
    name = file_name.removeprefix("telemetry_").removesuffix("_clean_1hz.csv")
    name = name.replace("mixed_normal_", "mixed_")
    name = name.replace("_normal", "")
    return name


def write_markdown_table(df: pd.DataFrame, path: Path, float_format: str = ".4f") -> None:
    def fmt(value):
        if isinstance(value, float):
            return format(value, float_format)
        return str(value)

    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_infected_table(infected_results: dict) -> pd.DataFrame:
    rows = []
    for item in infected_results["files"]:
        rows.append(
            {
                "file": short_name(item["file"]),
                "rows": item["rows"],
                "windows": item["windows"],
                "positive_windows": item["positive_windows"],
                "positive_rate": item["positive_window_rate"],
                "score_min": item["score_min"],
                "score_mean": item["score_mean"],
                "score_max": item["score_max"],
                "first_event_sec": item["first_event_start_sec"],
                "events": len(item["events"]),
            }
        )
    return pd.DataFrame(rows)


def build_mixed_table(mixed_results: dict) -> pd.DataFrame:
    rows = []
    for item in mixed_results["files"]:
        metrics = item["metrics"]
        cm = item["confusion_matrix"]
        rows.append(
            {
                "file": short_name(item["file"]),
                "rows": item["rows"],
                "windows": item["windows"],
                "TP": cm["TP"],
                "FP": cm["FP"],
                "FN": cm["FN"],
                "TN": cm["TN"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "FPR": metrics["false_positive_rate"],
                "positive_rate": item["positive_window_rate"],
                "first_event_sec": item["first_event_start_sec"],
                "delay_sec": item["detection_delay_sec"],
                "events": len(item["events"]),
            }
        )
    return pd.DataFrame(rows)


def build_summary_table(infected_df: pd.DataFrame, mixed_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "validation_normal",
                "files": "-",
                "threshold": threshold,
                "mean_precision": "-",
                "mean_recall": "-",
                "mean_f1": "-",
                "mean_FPR": "-",
                "mean_positive_rate": "-",
            },
            {
                "dataset": "infected_only",
                "files": len(infected_df),
                "threshold": threshold,
                "mean_precision": "-",
                "mean_recall": "-",
                "mean_f1": "-",
                "mean_FPR": "-",
                "mean_positive_rate": float(infected_df["positive_rate"].mean()),
            },
            {
                "dataset": "mixed",
                "files": len(mixed_df),
                "threshold": threshold,
                "mean_precision": float(mixed_df["precision"].mean()),
                "mean_recall": float(mixed_df["recall"].mean()),
                "mean_f1": float(mixed_df["f1_score"].mean()),
                "mean_FPR": float(mixed_df["FPR"].mean()),
                "mean_positive_rate": float(mixed_df["positive_rate"].mean()),
            },
        ]
    )


def save_tables(infected_df: pd.DataFrame, mixed_df: pd.DataFrame, summary_df: pd.DataFrame) -> list[Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    outputs = []
    for name, df in [
        ("infected_metrics", infected_df),
        ("mixed_metrics", mixed_df),
        ("summary_metrics", summary_df),
    ]:
        csv_path = TABLE_DIR / f"{name}.csv"
        md_path = TABLE_DIR / f"{name}.md"
        df.to_csv(csv_path, index=False)
        write_markdown_table(df, md_path)
        outputs.extend([csv_path, md_path])

    return outputs


def plot_validation_distribution(threshold: float) -> Path:
    val_data = torch.load(VAL_SCORES_PATH, map_location="cpu", weights_only=False)
    val_score = to_numpy(val_data[VAL_SCORE_KEY])

    out_path = FIG_DIR / "validation_score_distribution.png"
    plt.figure(figsize=(10, 5))
    plt.hist(val_score, bins=50)
    plt.axvline(threshold, linestyle="--", color="red", label=f"threshold={threshold:.4f}")
    plt.title("Validation normal score distribution")
    plt.xlabel(VAL_SCORE_KEY)
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def collect_scores(results: dict, score_key: str) -> np.ndarray:
    chunks = []
    for item in results["files"]:
        data = torch.load(scores_path(item["scores_path"]), map_location="cpu", weights_only=False)
        chunks.append(to_numpy(data[score_key]))
    return np.concatenate(chunks)


def plot_normal_vs_infected_distribution(
    threshold: float,
    infected_results: dict,
    mixed_results: dict,
) -> Path:
    val_data = torch.load(VAL_SCORES_PATH, map_location="cpu", weights_only=False)
    val_score = np.clip(to_numpy(val_data[VAL_SCORE_KEY]), 1e-8, None)
    infected_score = np.clip(collect_scores(infected_results, infected_results["score_key"]), 1e-8, None)
    mixed_score = np.clip(collect_scores(mixed_results, mixed_results["score_key"]), 1e-8, None)

    max_score = max(val_score.max(), infected_score.max(), mixed_score.max())
    min_score = min(val_score.min(), infected_score.min(), mixed_score.min())
    bins = np.logspace(np.log10(min_score), np.log10(max_score), 80)

    out_path = FIG_DIR / "score_distribution_normal_infected_mixed_log.png"
    plt.figure(figsize=(11, 6))
    plt.hist(val_score, bins=bins, alpha=0.65, label="validation normal")
    plt.hist(infected_score, bins=bins, alpha=0.55, label="infected only")
    plt.hist(mixed_score, bins=bins, alpha=0.45, label="mixed")
    plt.axvline(max(threshold, 1e-8), linestyle="--", color="red", label=f"threshold={threshold:.4f}")
    plt.xscale("log")
    plt.title("Anomaly score distributions")
    plt.xlabel("score, log scale")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_mixed_timeline(item: dict, mixed_results: dict) -> Path:
    score_key = mixed_results["score_key"]
    threshold = mixed_results["threshold"]
    stride = mixed_results["window_config"]["stride"]
    true_interval = mixed_results["true_infected_interval_sec"]

    data = torch.load(scores_path(item["scores_path"]), map_location="cpu", weights_only=False)
    score = np.clip(to_numpy(data[score_key]), 1e-8, None)
    x = np.arange(len(score)) * stride

    out_path = FIG_DIR / f"timeline_{Path(item['file']).stem}.png"
    plt.figure(figsize=(13, 5))
    plt.plot(x, score, label=score_key)
    plt.axhline(max(threshold, 1e-8), linestyle="--", color="red", label=f"threshold={threshold:.4f}")
    plt.axvspan(
        true_interval["start"],
        true_interval["end"],
        alpha=0.2,
        color="orange",
        label="true infected interval",
    )

    if item["first_event_start_sec"] is not None:
        first_event = item["events"][0]
        plt.axvspan(
            first_event["start_sec"],
            first_event["end_sec"],
            alpha=0.15,
            color="green",
            label="first detected event",
        )

    plt.yscale("log")
    plt.title(short_name(item["file"]))
    plt.xlabel("time, seconds")
    plt.ylabel("score, log scale")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_mixed_metrics(mixed_df: pd.DataFrame) -> Path:
    out_path = FIG_DIR / "mixed_metrics_bar.png"
    labels = mixed_df["file"].tolist()
    x = np.arange(len(labels))
    width = 0.22

    plt.figure(figsize=(13, 6))
    plt.bar(x - width, mixed_df["precision"], width, label="precision")
    plt.bar(x, mixed_df["recall"], width, label="recall")
    plt.bar(x + width, mixed_df["f1_score"], width, label="F1")
    plt.ylim(0, 1.05)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("metric value")
    plt.title("Mixed scenario metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_confusion_matrix_total(mixed_results: dict) -> Path:
    total = np.zeros((2, 2), dtype=int)
    for item in mixed_results["files"]:
        cm = item["confusion_matrix"]
        total += np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]], dtype=int)

    out_path = FIG_DIR / "mixed_confusion_matrix_total.png"
    plt.figure(figsize=(5, 4))
    plt.imshow(total, cmap="Blues")
    plt.title("Mixed total confusion matrix")
    plt.xticks([0, 1], ["pred normal", "pred infected"])
    plt.yticks([0, 1], ["true normal", "true infected"])

    for y in range(2):
        for x in range(2):
            plt.text(x, y, str(total[y, x]), ha="center", va="center", color="black")

    plt.colorbar(label="windows")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def save_figures(
    threshold: float,
    infected_results: dict,
    mixed_results: dict,
    mixed_df: pd.DataFrame,
) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    outputs = [
        plot_validation_distribution(threshold),
        plot_normal_vs_infected_distribution(threshold, infected_results, mixed_results),
        plot_mixed_metrics(mixed_df),
        plot_confusion_matrix_total(mixed_results),
    ]

    for item in mixed_results["files"]:
        outputs.append(plot_mixed_timeline(item, mixed_results))

    return outputs


def main() -> None:
    stats = load_json(STATS_PATH)
    infected_results = load_json(INFECTED_RESULTS_PATH)
    mixed_results = load_json(MIXED_RESULTS_PATH)
    threshold = stats["recommended_threshold"]

    infected_df = build_infected_table(infected_results)
    mixed_df = build_mixed_table(mixed_results)
    summary_df = build_summary_table(infected_df, mixed_df, threshold)

    table_outputs = save_tables(infected_df, mixed_df, summary_df)
    figure_outputs = save_figures(threshold, infected_results, mixed_results, mixed_df)

    print("=== Report artifacts ===")
    print("Tables:")
    for path in table_outputs:
        print(" -", path)
    print()
    print("Figures:")
    for path in figure_outputs:
        print(" -", path)


if __name__ == "__main__":
    main()
