import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model_scripts.score_windows import TCNForecaster, compute_basic_scores
from data_scripts.step3_make_windows import make_windows
from project_config import DATASET_PATH, H, MIN_CONSECUTIVE_WINDOWS, S, SCORE_KEY, T, TIME_COL


BASE_DIR = PROJECT_ROOT
CLEAN_INFECTED_DIR = BASE_DIR / "clean_data" / "infected"
CLEAN_NORMAL_DIR = BASE_DIR / "clean_data" / "normal"
BEST_PATH = BASE_DIR / "checkpoints" / "tcn_best.pt"
BASELINE_SCORES_PATH = BASE_DIR / "checkpoints" / "val_scores_from_script.pt"
STATS_PATH = BASE_DIR / "checkpoints" / "score_stats.json"
OUT_DIR = BASE_DIR / "checkpoints" / "infected_dataset"
RESULTS_PATH = BASE_DIR / "results" / "infected_dataset_evaluation.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mean_topk(x: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, x.shape[1])
    topk_vals, _ = torch.topk(x, k=k, dim=1)
    return topk_vals.mean(dim=1)


def load_reference_columns() -> list[str]:
    normal_files = sorted(CLEAN_NORMAL_DIR.glob("*.csv"))
    if not normal_files:
        raise FileNotFoundError(f"No normal cleaned files in {CLEAN_NORMAL_DIR}")

    ref = pd.read_csv(normal_files[0], nrows=1)
    return [c for c in ref.columns if c != TIME_COL]


def load_feature_frame(csv_path: Path, reference_cols: list[str]) -> tuple[pd.Series, np.ndarray]:
    df = pd.read_csv(csv_path)
    if TIME_COL not in df.columns:
        raise ValueError(f"{csv_path.name}: missing required column {TIME_COL}")

    current_cols = [c for c in df.columns if c != TIME_COL]
    missing = sorted(set(reference_cols) - set(current_cols))
    extra = sorted(set(current_cols) - set(reference_cols))
    if missing or extra:
        raise ValueError(
            f"{csv_path.name}: feature columns do not match training data. "
            f"Missing={missing}; extra={extra}"
        )

    ts_values = df[TIME_COL].copy()
    X = df[reference_cols].to_numpy(dtype=np.float32)
    return ts_values, X


def find_consecutive_runs(flags: np.ndarray) -> list[tuple[int, int, int]]:
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


def score_file(
    csv_path: Path,
    reference_cols: list[str],
    model: TCNForecaster,
    mu: np.ndarray,
    sigma: np.ndarray,
    feature_err_mean: torch.Tensor,
    feature_err_std: torch.Tensor,
    threshold: float,
) -> dict:
    ts_values, X = load_feature_frame(csv_path, reference_cols)
    X = (X - mu) / sigma

    X_past, Y_future = make_windows(X, T, H, S)
    if X_past.shape[0] == 0:
        raise ValueError(f"{csv_path.name}: not enough rows to create windows")

    xb = torch.from_numpy(X_past.astype(np.float32)).to(DEVICE)
    yb = torch.from_numpy(Y_future.astype(np.float32)).to(DEVICE)

    with torch.no_grad():
        xb_tcn = xb.permute(0, 2, 1).contiguous()
        yhat = model(xb_tcn)
        feature_mse, global_mse, vector_l2 = compute_basic_scores(yhat, yb)

    feature_mse = feature_mse.cpu()
    global_mse = global_mse.cpu()
    vector_l2 = vector_l2.cpu()

    z_feature = (feature_mse - feature_err_mean) / (feature_err_std + 1e-8)
    z_feature_pos = torch.clamp(z_feature, min=0.0)

    scores = {
        "std_score_mean": z_feature_pos.mean(dim=1),
        "std_score_max": z_feature_pos.max(dim=1).values,
        "std_score_top5": mean_topk(z_feature_pos, k=5),
    }
    score = scores[SCORE_KEY].numpy()
    predicted_positive = score > threshold
    event_runs = [
        run for run in find_consecutive_runs(predicted_positive)
        if run[2] >= MIN_CONSECUTIVE_WINDOWS
    ]

    out_path = OUT_DIR / f"{csv_path.stem}_scores.pt"
    torch.save(
        {
            "feature_mse": feature_mse,
            "global_mse": global_mse,
            "vector_l2": vector_l2,
            "z_feature": z_feature,
            "z_feature_pos": z_feature_pos,
            **scores,
            "window_timestamps": ts_values.iloc[::S].head(len(score)).tolist(),
            "infected_file": csv_path.name,
            "feature_columns": reference_cols,
        },
        out_path,
    )

    first_event_start_sec = event_runs[0][0] * S if event_runs else None
    return {
        "file": csv_path.name,
        "scores_path": str(out_path.relative_to(BASE_DIR)),
        "rows": int(X.shape[0]),
        "windows": int(len(score)),
        "score_key": SCORE_KEY,
        "threshold": float(threshold),
        "score_min": float(score.min()),
        "score_max": float(score.max()),
        "score_mean": float(score.mean()),
        "positive_windows": int(predicted_positive.sum()),
        "positive_window_rate": float(predicted_positive.mean()),
        "first_event_start_sec": first_event_start_sec,
        "events": [
            {
                "start_window": int(start),
                "end_window": int(end),
                "length_windows": int(length),
                "start_sec": int(start * S),
                "end_sec": int(end * S),
            }
            for start, end, length in event_runs
        ],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(CLEAN_INFECTED_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No infected cleaned files in {CLEAN_INFECTED_DIR}")

    reference_cols = load_reference_columns()
    data = np.load(DATASET_PATH, allow_pickle=False)
    mu = data["mu"]
    sigma = data["sigma"]

    baseline = torch.load(BASELINE_SCORES_PATH, map_location="cpu", weights_only=False)
    feature_err_mean = baseline["feature_err_mean"]
    feature_err_std = baseline["feature_err_std"]
    if not isinstance(feature_err_mean, torch.Tensor):
        feature_err_mean = torch.tensor(feature_err_mean, dtype=torch.float32)
    if not isinstance(feature_err_std, torch.Tensor):
        feature_err_std = torch.tensor(feature_err_std, dtype=torch.float32)

    with open(STATS_PATH, "r", encoding="utf-8") as f:
        threshold = json.load(f)["recommended_threshold"]

    model = TCNForecaster(in_dim=len(reference_cols), hidden_ch=64, H=H).to(DEVICE)
    ckpt = torch.load(BEST_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loaded checkpoint epoch={ckpt['epoch']} val_loss={ckpt['val_loss']:.6f}")
    print(f"Scoring {len(files)} infected cleaned file(s)")
    print(f"Threshold ({SCORE_KEY}): {threshold}")
    print()

    summaries = []
    for csv_path in files:
        summary = score_file(
            csv_path=csv_path,
            reference_cols=reference_cols,
            model=model,
            mu=mu,
            sigma=sigma,
            feature_err_mean=feature_err_mean,
            feature_err_std=feature_err_std,
            threshold=threshold,
        )
        summaries.append(summary)
        print(
            f"{summary['file']}: windows={summary['windows']} "
            f"positive={summary['positive_windows']} "
            f"rate={summary['positive_window_rate']:.3f} "
            f"max={summary['score_max']:.3f} "
            f"first_event_sec={summary['first_event_start_sec']}"
        )

    results = {
        "score_key": SCORE_KEY,
        "threshold": float(threshold),
        "window_config": {"T": T, "H": H, "stride": S},
        "min_consecutive_windows_for_event": MIN_CONSECUTIVE_WINDOWS,
        "files": summaries,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print()
    print("Saved infected dataset evaluation to:", RESULTS_PATH)


if __name__ == "__main__":
    main()
