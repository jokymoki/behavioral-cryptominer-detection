import argparse
import json
import random
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


EXP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXP_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from project_config import NORMAL_HOLDOUT_FILES

CLEAN_NORMAL_DIR = PROJECT_DIR / "clean_data" / "normal"
CLEAN_INFECTED_DIR = PROJECT_DIR / "clean_data" / "infected"
CLEAN_MIXED_DIR = PROJECT_DIR / "clean_data" / "mixed"

RUNS_DIR = EXP_DIR / "runs"
TABLE_DIR = EXP_DIR / "tables"
FIG_DIR = EXP_DIR / "figures"

TIME_COL = "ts"
T = 120
S = 10
DEFAULT_HORIZONS = [1, 5, 10, 20, 30]
TRUE_INFECTED_START_SEC = 2000
TRUE_INFECTED_END_SEC = 4000
MIN_CONSECUTIVE = 3
SCORE_KEY = "std_score_top5"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_reference_columns() -> list[str]:
    files = sorted(CLEAN_NORMAL_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No normal CSV files in {CLEAN_NORMAL_DIR}")
    df = pd.read_csv(files[0], nrows=1)
    return [c for c in df.columns if c != TIME_COL]


def validate_columns(path: Path, reference_cols: list[str]) -> pd.DataFrame:
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


def make_windows(X: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    n, d = X.shape
    m = (n - (T + horizon)) // S + 1
    if m <= 0:
        return (
            np.zeros((0, T, d), dtype=np.float32),
            np.zeros((0, horizon, d), dtype=np.float32),
        )

    x_past = np.zeros((m, T, d), dtype=np.float32)
    y_future = np.zeros((m, horizon, d), dtype=np.float32)
    for i, start in enumerate(range(0, n - (T + horizon) + 1, S)):
        x_past[i] = X[start:start + T]
        y_future[i] = X[start + T:start + T + horizon]

    return x_past, y_future


def build_dataset(horizon: int, run_dir: Path, reference_cols: list[str]) -> dict:
    normal_files = [
        f for f in sorted(CLEAN_NORMAL_DIR.glob("*.csv"))
        if f.name not in NORMAL_HOLDOUT_FILES
    ]
    x_list = []
    y_list = []
    file_summaries = []

    for path in normal_files:
        df = validate_columns(path, reference_cols)
        X = df[reference_cols].to_numpy(dtype=np.float32)
        x_past, y_future = make_windows(X, horizon)
        if x_past.shape[0] > 0:
            x_list.append(x_past)
            y_list.append(y_future)
        file_summaries.append({"file": path.name, "rows": int(len(df)), "windows": int(x_past.shape[0])})

    if not x_list:
        raise ValueError("No training windows were created")

    x_all = np.concatenate(x_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    n = x_all.shape[0]
    n_train = int(n * 0.8)
    x_train = x_all[:n_train]
    y_train = y_all[:n_train]
    x_val = x_all[n_train:]
    y_val = y_all[n_train:]

    mu = x_train.mean(axis=(0, 1))
    sigma = x_train.std(axis=(0, 1))
    sigma = np.where(sigma < 1e-8, 1.0, sigma)

    x_train = (x_train - mu) / sigma
    y_train = (y_train - mu) / sigma
    x_val = (x_val - mu) / sigma
    y_val = (y_val - mu) / sigma

    dataset_dir = run_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "windows.npz"
    np.savez_compressed(
        dataset_path,
        X_train=x_train.astype(np.float32),
        Y_train=y_train.astype(np.float32),
        X_val=x_val.astype(np.float32),
        Y_val=y_val.astype(np.float32),
        mu=mu.astype(np.float32),
        sigma=sigma.astype(np.float32),
    )

    summary = {
        "horizon": horizon,
        "T": T,
        "S": S,
        "excluded_holdout_normal_files": NORMAL_HOLDOUT_FILES,
        "normal_files": file_summaries,
        "total_windows": int(n),
        "train_windows": int(x_train.shape[0]),
        "val_windows": int(x_val.shape[0]),
        "features": int(x_train.shape[2]),
        "dataset_path": str(dataset_path.relative_to(EXP_DIR)),
    }
    (dataset_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.dropout(out)
        out = F.relu(self.conv2(out))
        out = self.dropout(out)
        return x + out


class TCNBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden_ch: int = 64, kernel_size: int = 3, levels: int = 6):
        super().__init__()
        self.in_proj = nn.Conv1d(in_dim, hidden_ch, kernel_size=1)
        self.blocks = nn.Sequential(
            *[TCNBlock(hidden_ch, kernel_size, 2 ** i) for i in range(levels)]
        )

    def forward(self, x):
        return self.blocks(self.in_proj(x))


class TCNForecaster(nn.Module):
    def __init__(self, in_dim: int, horizon: int, hidden_ch: int = 64):
        super().__init__()
        self.horizon = horizon
        self.backbone = TCNBackbone(in_dim, hidden_ch)
        self.head = nn.Linear(hidden_ch, horizon * in_dim)

    def forward(self, x):
        z = self.backbone(x)
        z_last = z[:, :, -1]
        y_flat = self.head(z_last)
        return y_flat.view(x.shape[0], self.horizon, -1)


def compute_basic_scores(pred: torch.Tensor, target: torch.Tensor):
    sq_err = (pred - target) ** 2
    feature_mse = sq_err.mean(dim=1)
    global_mse = feature_mse.mean(dim=1)
    vector_l2 = torch.sqrt((feature_mse ** 2).sum(dim=1))
    return feature_mse, global_mse, vector_l2


def mean_topk(x: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, x.shape[1])
    topk_vals, _ = torch.topk(x, k=k, dim=1)
    return topk_vals.mean(dim=1)


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion) -> float:
    model.eval()
    running = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            yhat = model(xb.permute(0, 2, 1).contiguous())
            running += criterion(yhat, yb).item()
    return running / max(1, len(loader))


def train_model(
    run_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
) -> dict:
    dataset = np.load(run_dir / "dataset" / "windows.npz", allow_pickle=False)
    x_train = dataset["X_train"]
    y_train = dataset["Y_train"]
    x_val = dataset["X_val"]
    y_val = dataset["Y_val"]

    train_loader = DataLoader(WindowDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    horizon = y_train.shape[1]
    in_dim = x_train.shape[2]
    model = TCNForecaster(in_dim=in_dim, horizon=horizon).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "tcn_best.pt"
    last_path = ckpt_dir / "tcn_last.pt"

    best_val = float("inf")
    last_val = None
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    best_epoch = None

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            yhat = model(xb.permute(0, 2, 1).contiguous())
            loss = criterion(yhat, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

        avg_train = running / max(1, len(train_loader))
        avg_val = evaluate_loss(model, val_loader, criterion)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        last_val = avg_val

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": avg_val,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
        torch.save(ckpt, last_path)

        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(ckpt, best_path)
        else:
            epochs_no_improve += 1

        print(
            f"  epoch={epoch:02d} train_loss={avg_train:.6f} "
            f"val_loss={avg_val:.6f} best={best_val:.6f}"
        )
        if epochs_no_improve >= patience:
            print(f"  early stopping at epoch {epoch}")
            break

    return {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "last_val_loss": float(last_val),
        "best_path": str(best_path.relative_to(EXP_DIR)),
    }


def score_validation(run_dir: Path) -> dict:
    dataset = np.load(run_dir / "dataset" / "windows.npz", allow_pickle=False)
    x_val = dataset["X_val"]
    y_val = dataset["Y_val"]

    model = load_model(run_dir, in_dim=x_val.shape[2], horizon=y_val.shape[1])
    loader = DataLoader(WindowDataset(x_val, y_val), batch_size=64, shuffle=False)

    feature_mse_chunks = []
    global_mse_chunks = []
    vector_l2_chunks = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            yhat = model(xb.permute(0, 2, 1).contiguous())
            feature_mse, global_mse, vector_l2 = compute_basic_scores(yhat, yb)
            feature_mse_chunks.append(feature_mse.cpu())
            global_mse_chunks.append(global_mse.cpu())
            vector_l2_chunks.append(vector_l2.cpu())

    feature_mse = torch.cat(feature_mse_chunks, dim=0)
    global_mse = torch.cat(global_mse_chunks, dim=0)
    vector_l2 = torch.cat(vector_l2_chunks, dim=0)

    feature_err_mean = feature_mse.mean(dim=0)
    feature_err_std_raw = feature_mse.std(dim=0)
    feature_err_std = torch.clamp(feature_err_std_raw, min=0.05)

    z_feature = (feature_mse - feature_err_mean) / feature_err_std
    z_feature_pos = torch.clamp(z_feature, min=0.0)
    std_score = z_feature_pos.mean(dim=1)

    threshold = torch.quantile(std_score, 0.99).item()
    stats = {
        "num_val_windows": int(std_score.shape[0]),
        "std_score_mean": float(std_score.mean().item()),
        "std_score_std": float(std_score.std().item()),
        "std_score_min": float(std_score.min().item()),
        "std_score_max": float(std_score.max().item()),
        "std_score_p95": float(torch.quantile(std_score, 0.95).item()),
        "std_score_p99": float(threshold),
        "recommended_threshold": float(threshold),
    }

    ckpt_dir = run_dir / "checkpoints"
    torch.save(
        {
            "feature_mse": feature_mse,
            "global_mse": global_mse,
            "vector_l2": vector_l2,
            "std_score": std_score,
            "feature_err_mean": feature_err_mean,
            "feature_err_std": feature_err_std,
            "feature_err_std_raw": feature_err_std_raw,
        },
        ckpt_dir / "val_scores.pt",
    )
    (ckpt_dir / "score_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def load_model(run_dir: Path, in_dim: int, horizon: int) -> TCNForecaster:
    model = TCNForecaster(in_dim=in_dim, horizon=horizon).to(DEVICE)
    ckpt = torch.load(run_dir / "checkpoints" / "tcn_best.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


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


def build_true_labels(num_windows: int, horizon: int) -> np.ndarray:
    labels = np.zeros(num_windows, dtype=bool)
    for i in range(num_windows):
        window_start = i * S
        window_end = window_start + T + horizon
        labels[i] = (
            window_start < TRUE_INFECTED_END_SEC
            and window_end > TRUE_INFECTED_START_SEC
        )
    return labels


def score_csv(
    path: Path,
    reference_cols: list[str],
    run_dir: Path,
    horizon: int,
    model: TCNForecaster,
    mu: np.ndarray,
    sigma: np.ndarray,
    baseline: dict,
    threshold: float,
) -> tuple[dict, dict]:
    df = validate_columns(path, reference_cols)
    X = df[reference_cols].to_numpy(dtype=np.float32)
    X = (X - mu) / sigma
    x_past, y_future = make_windows(X, horizon)
    if x_past.shape[0] == 0:
        raise ValueError(f"{path.name}: not enough rows for H={horizon}")

    xb = torch.from_numpy(x_past.astype(np.float32)).to(DEVICE)
    yb = torch.from_numpy(y_future.astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        yhat = model(xb.permute(0, 2, 1).contiguous())
        feature_mse, global_mse, vector_l2 = compute_basic_scores(yhat, yb)

    feature_mse = feature_mse.cpu()
    z_feature = (feature_mse - baseline["feature_err_mean"]) / (baseline["feature_err_std"] + 1e-8)
    z_feature_pos = torch.clamp(z_feature, min=0.0)
    scores = {
        "std_score_mean": z_feature_pos.mean(dim=1),
        "std_score_max": z_feature_pos.max(dim=1).values,
        "std_score_top5": mean_topk(z_feature_pos, k=5),
    }
    score = scores[SCORE_KEY].numpy()
    predicted = score > threshold

    score_payload = {
        "feature_mse": feature_mse,
        "global_mse": global_mse.cpu(),
        "vector_l2": vector_l2.cpu(),
        "z_feature": z_feature,
        "z_feature_pos": z_feature_pos,
        **scores,
        "file": path.name,
        "horizon": horizon,
    }

    summary = {
        "file": path.name,
        "rows": int(len(df)),
        "windows": int(len(score)),
        "score_min": float(score.min()),
        "score_mean": float(score.mean()),
        "score_max": float(score.max()),
        "positive_windows": int(predicted.sum()),
        "positive_window_rate": float(predicted.mean()),
    }
    return summary, score_payload


def evaluate_infected(run_dir: Path, reference_cols: list[str], horizon: int) -> dict:
    dataset = np.load(run_dir / "dataset" / "windows.npz", allow_pickle=False)
    mu = dataset["mu"]
    sigma = dataset["sigma"]
    model = load_model(run_dir, in_dim=len(reference_cols), horizon=horizon)
    baseline = torch.load(run_dir / "checkpoints" / "val_scores.pt", map_location="cpu", weights_only=False)
    stats = json.loads((run_dir / "checkpoints" / "score_stats.json").read_text(encoding="utf-8"))
    threshold = stats["recommended_threshold"]

    scores_dir = run_dir / "scores" / "infected"
    scores_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in sorted(CLEAN_INFECTED_DIR.glob("*.csv")):
        summary, payload = score_csv(
            path, reference_cols, run_dir, horizon, model, mu, sigma, baseline, threshold
        )
        score = payload[SCORE_KEY].numpy()
        predicted = score > threshold
        events = [run for run in find_runs(predicted) if run[2] >= MIN_CONSECUTIVE]
        summary["first_event_start_sec"] = int(events[0][0] * S) if events else None
        summary["events"] = [
            {
                "start_window": int(start),
                "end_window": int(end),
                "length_windows": int(length),
                "start_sec": int(start * S),
                "end_sec": int(end * S),
            }
            for start, end, length in events
        ]
        out_path = scores_dir / f"{path.stem}_scores.pt"
        torch.save(payload, out_path)
        summary["scores_path"] = str(out_path.relative_to(EXP_DIR))
        summaries.append(summary)

    result = {
        "horizon": horizon,
        "score_key": SCORE_KEY,
        "threshold": float(threshold),
        "files": summaries,
    }
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "infected_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


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


def evaluate_mixed(run_dir: Path, reference_cols: list[str], horizon: int) -> dict:
    dataset = np.load(run_dir / "dataset" / "windows.npz", allow_pickle=False)
    mu = dataset["mu"]
    sigma = dataset["sigma"]
    model = load_model(run_dir, in_dim=len(reference_cols), horizon=horizon)
    baseline = torch.load(run_dir / "checkpoints" / "val_scores.pt", map_location="cpu", weights_only=False)
    stats = json.loads((run_dir / "checkpoints" / "score_stats.json").read_text(encoding="utf-8"))
    threshold = stats["recommended_threshold"]

    scores_dir = run_dir / "scores" / "mixed"
    scores_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in sorted(CLEAN_MIXED_DIR.glob("*.csv")):
        summary, payload = score_csv(
            path, reference_cols, run_dir, horizon, model, mu, sigma, baseline, threshold
        )
        score = payload[SCORE_KEY].numpy()
        predicted = score > threshold
        true_labels = build_true_labels(len(score), horizon)
        cm = confusion(predicted, true_labels)
        events = [run for run in find_runs(predicted) if run[2] >= MIN_CONSECUTIVE]
        first_event_start_sec = int(events[0][0] * S) if events else None

        summary["confusion_matrix"] = {k: cm[k] for k in ["TP", "FP", "FN", "TN"]}
        summary["metrics"] = {k: cm[k] for k in ["precision", "recall", "f1_score", "false_positive_rate"]}
        summary["first_event_start_sec"] = first_event_start_sec
        summary["detection_delay_sec"] = (
            first_event_start_sec - TRUE_INFECTED_START_SEC
            if first_event_start_sec is not None
            else None
        )
        summary["events"] = [
            {
                "start_window": int(start),
                "end_window": int(end),
                "length_windows": int(length),
                "start_sec": int(start * S),
                "end_sec": int(end * S),
            }
            for start, end, length in events
        ]
        out_path = scores_dir / f"{path.stem}_scores.pt"
        torch.save(payload, out_path)
        summary["scores_path"] = str(out_path.relative_to(EXP_DIR))
        summaries.append(summary)

    aggregate = {
        "mean_precision": float(np.mean([x["metrics"]["precision"] for x in summaries])),
        "mean_recall": float(np.mean([x["metrics"]["recall"] for x in summaries])),
        "mean_f1_score": float(np.mean([x["metrics"]["f1_score"] for x in summaries])),
        "mean_false_positive_rate": float(np.mean([x["metrics"]["false_positive_rate"] for x in summaries])),
        "mean_detection_delay_sec": float(np.mean([x["detection_delay_sec"] for x in summaries])),
    }
    result = {
        "horizon": horizon,
        "score_key": SCORE_KEY,
        "threshold": float(threshold),
        "window_config": {"T": T, "H": horizon, "stride": S},
        "true_infected_interval_sec": {
            "start": TRUE_INFECTED_START_SEC,
            "end": TRUE_INFECTED_END_SEC,
        },
        "aggregate_metrics": aggregate,
        "files": summaries,
    }
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "mixed_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


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


def save_aggregate_outputs(rows: list[dict]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows).sort_values("H")
    csv_path = TABLE_DIR / "horizon_ablation_summary.csv"
    md_path = TABLE_DIR / "horizon_ablation_summary.md"
    df.to_csv(csv_path, index=False)
    write_markdown_table(df, md_path)

    x = df["H"].to_numpy()

    plt.figure(figsize=(9, 5))
    plt.plot(x, df["mixed_mean_precision"], marker="o", label="precision")
    plt.plot(x, df["mixed_mean_recall"], marker="o", label="recall")
    plt.plot(x, df["mixed_mean_f1"], marker="o", label="F1")
    plt.xlabel("Forecast horizon H, seconds")
    plt.ylabel("metric")
    plt.ylim(0, 1.05)
    plt.title("Horizon ablation: mixed metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "horizon_ablation_metrics.png", dpi=300)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(x, df["mixed_mean_FPR"], marker="o", color="tab:red", label="FPR")
    ax1.set_xlabel("Forecast horizon H, seconds")
    ax1.set_ylabel("FPR", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(x, df["mixed_mean_detection_delay_sec"], marker="o", color="tab:blue", label="delay")
    ax2.set_ylabel("Detection delay, seconds", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    plt.title("Horizon ablation: FPR and detection delay")
    fig.tight_layout()
    plt.savefig(FIG_DIR / "horizon_ablation_fpr_delay.png", dpi=300)
    plt.close()

    print("Aggregate outputs:")
    print(" -", csv_path)
    print(" -", md_path)
    print(" -", FIG_DIR / "horizon_ablation_metrics.png")
    print(" -", FIG_DIR / "horizon_ablation_fpr_delay.png")


def dry_run(horizons: list[int]) -> None:
    reference_cols = read_reference_columns()
    normal_files = [
        f for f in sorted(CLEAN_NORMAL_DIR.glob("*.csv"))
        if f.name not in NORMAL_HOLDOUT_FILES
    ]
    infected_files = sorted(CLEAN_INFECTED_DIR.glob("*.csv"))
    mixed_files = sorted(CLEAN_MIXED_DIR.glob("*.csv"))

    print("Horizon ablation dry run")
    print("Project dir:", PROJECT_DIR)
    print("Experiment dir:", EXP_DIR)
    print("Device:", DEVICE)
    print("Horizons:", horizons)
    print("T:", T, "S:", S)
    print("Feature count:", len(reference_cols))
    print("Normal files:", len(normal_files))
    print("Excluded holdout normal files:", NORMAL_HOLDOUT_FILES)
    print("Infected files:", len(infected_files))
    print("Mixed files:", len(mixed_files))
    print()

    for horizon in horizons:
        total_windows = 0
        for path in normal_files:
            df = validate_columns(path, reference_cols)
            total_windows += max(0, (len(df) - (T + horizon)) // S + 1)
        print(f"H={horizon}: normal windows={total_windows}, train={int(total_windows * 0.8)}, val={total_windows - int(total_windows * 0.8)}")


def run(args) -> None:
    set_seed(args.seed)
    horizons = args.horizons
    reference_cols = read_reference_columns()

    if args.dry_run:
        dry_run(horizons)
        return

    aggregate_rows = []
    for horizon in horizons:
        run_name = f"H{horizon:03d}"
        run_dir = RUNS_DIR / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        print()
        print("=" * 80)
        print(f"Horizon H={horizon}")
        print("=" * 80)

        dataset_summary = build_dataset(horizon, run_dir, reference_cols)
        print(
            f"Dataset: total={dataset_summary['total_windows']} "
            f"train={dataset_summary['train_windows']} val={dataset_summary['val_windows']}"
        )

        train_summary = train_model(
            run_dir=run_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
        )
        val_stats = score_validation(run_dir)
        infected_results = evaluate_infected(run_dir, reference_cols, horizon)
        mixed_results = evaluate_mixed(run_dir, reference_cols, horizon)

        infected_positive_rate = float(
            np.mean([x["positive_window_rate"] for x in infected_results["files"]])
        )
        mixed_agg = mixed_results["aggregate_metrics"]

        row = {
            "H": horizon,
            "train_windows": dataset_summary["train_windows"],
            "val_windows": dataset_summary["val_windows"],
            "best_epoch": train_summary["best_epoch"],
            "best_val_loss": train_summary["best_val_loss"],
            "threshold": val_stats["recommended_threshold"],
            "infected_mean_positive_rate": infected_positive_rate,
            "mixed_mean_precision": mixed_agg["mean_precision"],
            "mixed_mean_recall": mixed_agg["mean_recall"],
            "mixed_mean_f1": mixed_agg["mean_f1_score"],
            "mixed_mean_FPR": mixed_agg["mean_false_positive_rate"],
            "mixed_mean_detection_delay_sec": mixed_agg["mean_detection_delay_sec"],
        }
        aggregate_rows.append(row)

        print("Horizon result:")
        print(
            f"  infected_positive_rate={infected_positive_rate:.4f} "
            f"mixed_precision={row['mixed_mean_precision']:.4f} "
            f"mixed_recall={row['mixed_mean_recall']:.4f} "
            f"mixed_F1={row['mixed_mean_f1']:.4f} "
            f"mixed_FPR={row['mixed_mean_FPR']:.4f} "
            f"delay={row['mixed_mean_detection_delay_sec']:.1f}s"
        )

    save_aggregate_outputs(aggregate_rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Run isolated horizon ablation experiment.")
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
