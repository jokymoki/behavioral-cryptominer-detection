import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


EXP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXP_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from project_config import (
    DATASET_PATH,
    H,
    MIN_CONSECUTIVE_WINDOWS,
    S,
    SCORE_KEY,
    T,
    TIME_COL,
    TRUE_INFECTED_END_SEC,
    TRUE_INFECTED_START_SEC,
    get_feature_columns,
)
from model_scripts.score_mixed_dataset import TCNForecaster, compute_basic_scores, mean_topk
from data_scripts.step3_make_windows import make_windows


CLEAN_NORMAL_DIR = PROJECT_DIR / "clean_data" / "normal"
CLEAN_INFECTED_DIR = PROJECT_DIR / "clean_data" / "infected"
DL_CKPT_PATH = PROJECT_DIR / "checkpoints" / "tcn_best.pt"
DL_BASELINE_PATH = PROJECT_DIR / "checkpoints" / "val_scores_from_script.pt"
DL_STATS_PATH = PROJECT_DIR / "checkpoints" / "score_stats.json"

CLASSICAL_MODEL_DIR = PROJECT_DIR / "experiments" / "classical_baselines" / "models"

GENERATED_DIR = EXP_DIR / "generated"
RESULTS_DIR = EXP_DIR / "results"
TABLE_DIR = EXP_DIR / "tables"
FIG_DIR = EXP_DIR / "figures"

NORMAL_FILE = "telemetry_Spotify_clean_1hz.csv"
INFECTED_FILE = "telemetry_infected_sample_tag_monero_clean_1hz.csv"

NORMAL_BEFORE_ROWS = 2000
INFECTED_ROWS = 2000
NORMAL_AFTER_ROWS = 2000
ALPHAS = [1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def reference_columns() -> list[str]:
    files = sorted(CLEAN_NORMAL_DIR.glob("*.csv"))
    df = pd.read_csv(files[0], nrows=1)
    return get_feature_columns(df.columns)


def take_rows_with_wrap(df: pd.DataFrame, start: int, count: int) -> pd.DataFrame:
    parts = []
    remaining = count
    current = start
    while remaining > 0:
        wrapped_start = current % len(df)
        chunk_size = min(remaining, len(df) - wrapped_start)
        parts.append(df.iloc[wrapped_start:wrapped_start + chunk_size])
        remaining -= chunk_size
        current += chunk_size
    return pd.concat(parts, axis=0, ignore_index=True).copy()


def build_stealth_mixed(alpha: float, cols: list[str]) -> Path:
    normal = pd.read_csv(CLEAN_NORMAL_DIR / NORMAL_FILE)
    infected = pd.read_csv(CLEAN_INFECTED_DIR / INFECTED_FILE)

    normal_before = take_rows_with_wrap(normal, 0, NORMAL_BEFORE_ROWS)
    normal_background = take_rows_with_wrap(normal, NORMAL_BEFORE_ROWS, INFECTED_ROWS)
    infected_part = infected.iloc[:INFECTED_ROWS].reset_index(drop=True).copy()
    normal_after = take_rows_with_wrap(normal, NORMAL_BEFORE_ROWS + INFECTED_ROWS, NORMAL_AFTER_ROWS)

    stealth_part = normal_background.copy()
    stealth_part[cols] = (
        normal_background[cols].to_numpy(dtype=np.float32)
        + alpha
        * (
            infected_part[cols].to_numpy(dtype=np.float32)
            - normal_background[cols].to_numpy(dtype=np.float32)
        )
    )

    mixed = pd.concat([normal_before, stealth_part, normal_after], axis=0, ignore_index=True)
    start_time = pd.to_datetime(normal[TIME_COL].iloc[0])
    mixed[TIME_COL] = pd.date_range(start=start_time, periods=len(mixed), freq="1s")

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GENERATED_DIR / f"stress_stealth_blend_alpha_{alpha:.2f}.csv"
    mixed.to_csv(out_path, index=False)
    return out_path


def true_labels(num_windows: int, window_span: int) -> np.ndarray:
    labels = np.zeros(num_windows, dtype=bool)
    for i in range(num_windows):
        window_start = i * S
        window_end = window_start + window_span
        labels[i] = (
            window_start < TRUE_INFECTED_END_SEC
            and window_end > TRUE_INFECTED_START_SEC
        )
    return labels


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


def metrics(predicted: np.ndarray, labels: np.ndarray) -> dict:
    tp = int(np.logical_and(predicted, labels).sum())
    fp = int(np.logical_and(predicted, ~labels).sum())
    fn = int(np.logical_and(~predicted, labels).sum())
    tn = int(np.logical_and(~predicted, ~labels).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    fpr = safe_div(fp, fp + tn)
    events = [run for run in find_runs(predicted) if run[2] >= MIN_CONSECUTIVE_WINDOWS]
    first_event = int(events[0][0] * S) if events else None
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "false_positive_rate": float(fpr),
        "positive_rate": float(predicted.mean()),
        "first_event_start_sec": first_event,
        "detection_delay_sec": first_event - TRUE_INFECTED_START_SEC if first_event is not None else None,
    }


def load_dl_model(in_dim: int) -> TCNForecaster:
    model = TCNForecaster(in_dim=in_dim, hidden_ch=64, H=H).to(DEVICE)
    ckpt = torch.load(DL_CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def score_dl(path: Path, cols: list[str], model: TCNForecaster) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(DATASET_PATH, allow_pickle=False)
    mu = data["mu"]
    sigma = data["sigma"]
    baseline = torch.load(DL_BASELINE_PATH, map_location="cpu", weights_only=False)
    with open(DL_STATS_PATH, "r", encoding="utf-8") as f:
        threshold = json.load(f)["recommended_threshold"]

    df = pd.read_csv(path)
    X = df[cols].to_numpy(dtype=np.float32)
    X = (X - mu) / sigma
    x_past, y_future = make_windows(X, T, H, S)

    xb = torch.from_numpy(x_past.astype(np.float32)).to(DEVICE)
    yb = torch.from_numpy(y_future.astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        yhat = model(xb.permute(0, 2, 1).contiguous())
        feature_mse, _, _ = compute_basic_scores(yhat, yb)

    feature_err_mean = baseline["feature_err_mean"]
    feature_err_std = baseline["feature_err_std"]
    z_feature = (feature_mse.cpu() - feature_err_mean) / (feature_err_std + 1e-8)
    z_feature_pos = torch.clamp(z_feature, min=0.0)
    score = mean_topk(z_feature_pos, k=5).numpy()
    return score, score > threshold


def classical_window_features(X: np.ndarray) -> np.ndarray:
    rows = []
    for start in range(0, len(X) - T + 1, S):
        w = X[start:start + T]
        rows.append(
            np.concatenate(
                [
                    w.mean(axis=0),
                    w.std(axis=0),
                    w.min(axis=0),
                    w.max(axis=0),
                    w[-1] - w[0],
                ]
            )
        )
    return np.asarray(rows, dtype=np.float32)


def score_classical(path: Path, cols: list[str], model_name: str) -> tuple[np.ndarray, np.ndarray]:
    model = joblib.load(CLASSICAL_MODEL_DIR / f"{model_name}.joblib")
    results = json.loads(
        (PROJECT_DIR / "experiments" / "classical_baselines" / "results" / "classical_baselines_results.json")
        .read_text(encoding="utf-8")
    )
    threshold = results["models"][model_name]["threshold"]

    df = pd.read_csv(path)
    X = df[cols].to_numpy(dtype=np.float32)
    F = classical_window_features(X)
    score = -model.decision_function(F)
    return score, score > threshold


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            vals.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figures(df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    for model_name, sub in df.groupby("model"):
        plt.plot(sub["alpha"], sub["recall"], marker="o", label=f"{model_name} recall")
    plt.gca().invert_xaxis()
    plt.xlabel("blend alpha: lower means stealthier")
    plt.ylabel("recall")
    plt.ylim(0, 1.05)
    plt.title("Stress stealth blend: recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "stress_mixed_recall.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    for model_name, sub in df.groupby("model"):
        plt.plot(sub["alpha"], sub["f1_score"], marker="o", label=f"{model_name} F1")
    plt.gca().invert_xaxis()
    plt.xlabel("blend alpha: lower means stealthier")
    plt.ylabel("F1")
    plt.ylim(0, 1.05)
    plt.title("Stress stealth blend: F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "stress_mixed_f1.png", dpi=300)
    plt.close()


def main() -> None:
    for directory in [GENERATED_DIR, RESULTS_DIR, TABLE_DIR, FIG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    cols = reference_columns()
    dl_model = load_dl_model(len(cols))
    rows = []
    details = []

    print("=== Stress mixed stealth blend ===")
    print(f"T={T}, H={H}, stride={S}")
    print(f"normal={NORMAL_FILE}")
    print(f"infected={INFECTED_FILE}")
    print()

    for alpha in ALPHAS:
        path = build_stealth_mixed(alpha, cols)

        dl_score, dl_pred = score_dl(path, cols, dl_model)
        dl_labels = true_labels(len(dl_pred), T + H)
        dl_metrics = metrics(dl_pred, dl_labels)
        rows.append({"scenario": path.name, "alpha": alpha, "model": "tcn_forecaster_dl", **dl_metrics})

        for model_name in ["isolation_forest", "one_class_svm"]:
            score, pred = score_classical(path, cols, model_name)
            labels = true_labels(len(pred), T + H)
            m = metrics(pred, labels)
            rows.append({"scenario": path.name, "alpha": alpha, "model": model_name, **m})

        details.append({"alpha": alpha, "csv_path": str(path.relative_to(EXP_DIR))})

    df = pd.DataFrame(rows)
    df = df.sort_values(["alpha", "model"], ascending=[False, True])

    csv_path = TABLE_DIR / "stress_mixed_summary.csv"
    md_path = TABLE_DIR / "stress_mixed_summary.md"
    df.to_csv(csv_path, index=False)
    write_markdown_table(df, md_path)
    save_figures(df)

    (RESULTS_DIR / "stress_mixed_results.json").write_text(
        json.dumps({"generated": details, "rows": rows}, indent=2),
        encoding="utf-8",
    )

    print(df[["alpha", "model", "precision", "recall", "f1_score", "false_positive_rate", "detection_delay_sec"]].to_string(index=False))
    print()
    print("Saved:")
    print(" -", csv_path)
    print(" -", md_path)
    print(" -", RESULTS_DIR / "stress_mixed_results.json")
    print(" -", FIG_DIR / "stress_mixed_recall.png")
    print(" -", FIG_DIR / "stress_mixed_f1.png")


if __name__ == "__main__":
    main()
