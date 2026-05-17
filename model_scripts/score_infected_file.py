import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from project_config import DATASET_PATH, H, S, T, TIME_COL

PROJECT_DIR = str(PROJECT_ROOT)
CLEAN_DIR = os.path.join(PROJECT_DIR, "clean_data", "mixed")
VAL_SCORES_PATH = os.path.join(PROJECT_DIR, "checkpoints", "val_scores_from_script.pt")
VAL_STATS_PATH = os.path.join(PROJECT_DIR, "checkpoints", "score_stats.json")
BEST_PATH = os.path.join(PROJECT_DIR, "checkpoints", "tcn_best.pt")
#OUT_PATH = os.path.join(PROJECT_DIR, "checkpoints", "infected_scores.pt")
OUT_PATH = os.path.join(PROJECT_DIR, "checkpoints", "mixed_scores.pt")
# put your cleaned infected filename here
#INFECTED_FILE = "telemetry_second_sample_of_virus_clean_1hz.csv"
INFECTED_FILE = "telemetry_mixed_normal_infected_normal_clean_1hz.csv"
OUT_PATH = os.path.join(PROJECT_DIR, "checkpoints", "mixed_scores.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_windows(X: np.ndarray, T: int, H: int, S: int):
    N, D = X.shape
    M = (N - (T + H)) // S + 1

    if M <= 0:
        return (
            np.zeros((0, T, D), dtype=np.float32),
            np.zeros((0, H, D), dtype=np.float32),
        )

    X_past = np.zeros((M, T, D), dtype=np.float32)
    Y_future = np.zeros((M, H, D), dtype=np.float32)

    i = 0
    for start in range(0, N - (T + H) + 1, S):
        X_past[i] = X[start:start + T]
        Y_future[i] = X[start + T:start + T + H]
        i += 1

    return X_past, Y_future


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = F.relu(out)
        out = self.dropout(out)

        return x + out


class TCNBackbone(nn.Module):
    def __init__(self, in_dim, hidden_ch=64, kernel_size=3, levels=6, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Conv1d(in_dim, hidden_ch, kernel_size=1)

        blocks = []
        for i in range(levels):
            d = 2 ** i
            blocks.append(TCNBlock(hidden_ch, kernel_size, d, dropout))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.blocks(x)
        return x


class TCNForecaster(nn.Module):
    def __init__(self, in_dim, hidden_ch, H):
        super().__init__()
        self.backbone = TCNBackbone(in_dim, hidden_ch)
        self.H = H
        self.head = nn.Linear(hidden_ch, H * in_dim)

    def forward(self, x):
        z = self.backbone(x)
        z_last = z[:, :, -1]
        y_flat = self.head(z_last)
        B = x.shape[0]
        y = y_flat.view(B, self.H, -1)
        return y


def compute_basic_scores(pred, target):
    err = pred - target
    sq_err = err ** 2

    feature_mse = sq_err.mean(dim=1)
    global_mse = feature_mse.mean(dim=1)
    vector_l2 = torch.sqrt((feature_mse ** 2).sum(dim=1))

    return feature_mse, global_mse, vector_l2

def mean_topk(x: torch.Tensor, k : int) :
    k = min(k, x.shape[1])
    topk_vals, _ = torch.topk(x, k=k, dim=1)
    return topk_vals.mean(dim=1)


def main():
    # 1. load infected cleaned csv
    csv_path = os.path.join(CLEAN_DIR, INFECTED_FILE)
    df = pd.read_csv(csv_path)

    if TIME_COL not in df.columns:
        raise ValueError(f"Missing column {TIME_COL}")

    ts_values = df[TIME_COL].copy()
    X = df.drop(columns=[TIME_COL]).to_numpy(dtype=np.float32)

    print("Loaded infected file:", csv_path)
    print("Raw infected shape:", X.shape)

    # 2. load train normalization stats
    data = np.load(DATASET_PATH, allow_pickle=False)
    mu = data["mu"]
    sigma = data["sigma"]

    X = (X - mu) / sigma

    # 3. make windows
    X_past, Y_future = make_windows(X, T, H, S)
    print("X_past shape:", X_past.shape)
    print("Y_future shape:", Y_future.shape)

    if X_past.shape[0] == 0:
        raise ValueError("Not enough rows to create windows")

    # 4. load model
    D = X_past.shape[2]
    model = TCNForecaster(in_dim=D, hidden_ch=64, H=H).to(DEVICE)

    ckpt = torch.load(BEST_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loaded checkpoint from epoch {ckpt['epoch']} with val_loss={ckpt['val_loss']:.6f}")

    # 5. score infected windows
    xb = torch.from_numpy(X_past.astype(np.float32)).to(DEVICE)
    yb = torch.from_numpy(Y_future.astype(np.float32)).to(DEVICE)

    with torch.no_grad():
        xb_tcn = xb.permute(0, 2, 1).contiguous()
        yhat = model(xb_tcn)

        feature_mse, global_mse, vector_l2 = compute_basic_scores(yhat, yb)

    feature_mse = feature_mse.cpu()
    global_mse = global_mse.cpu()
    vector_l2 = vector_l2.cpu()

    # 6. standardize 
    baseline = torch.load(VAL_SCORES_PATH, map_location="cpu", weights_only=False)

    feature_err_mean = baseline["feature_err_mean"]
    feature_err_std = baseline["feature_err_std"]

    if not isinstance(feature_err_mean, torch.Tensor):
        feature_err_mean = torch.tensor(feature_err_mean, dtype= torch.float32)
    
    if not isinstance(feature_err_std, torch.Tensor):
        feature_err_std = torch.tensor(feature_err_std, dtype=torch.float32)

    eps=1e-8

    z_feature = (feature_mse - feature_err_mean) / (feature_err_std + eps)
    z_feature_pos = torch.clamp(z_feature, min=0.0)

    std_score_mean = z_feature_pos.mean(dim=1)
    std_score_max = z_feature_pos.max(dim=1).values
    std_score_top5 = mean_topk(z_feature_pos, k=5)

    # 7. save results
    torch.save(
    {
        "feature_mse": feature_mse,
        "global_mse": global_mse,
        "vector_l2": vector_l2,
        "z_feature": z_feature,
        "z_feature_pos": z_feature_pos,
        "std_score_mean": std_score_mean,
        "std_score_max": std_score_max,
        "std_score_top5": std_score_top5,
        "window_timestamps": ts_values.tolist(),
        "infected_file": INFECTED_FILE,
        "feature_err_mean_normal": feature_err_mean,
        "feature_err_std_normal": feature_err_std,
    },
        OUT_PATH,
    )

    print("Saved infected scores to:", OUT_PATH)
    print("Num windows:", feature_mse.shape[0])

    print("std_score_mean min/max:", std_score_mean.min().item(), std_score_mean.max().item())
    print("std_score_max min/max:", std_score_max.min().item(), std_score_max.max().item())
    print("std_score_top5 min/max:", std_score_top5.min().item(), std_score_top5.max().item())


if __name__ == "__main__":
    main()
