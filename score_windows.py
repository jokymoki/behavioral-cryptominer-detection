import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(PROJECT_DIR, "datasets", "windows_T120_H10_S10.npz")
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
BEST_PATH = os.path.join(CKPT_DIR, "tcn_best.pt")

SCORES_OUT_PATH = os.path.join(CKPT_DIR, "val_scores_from_script.pt")
STATS_OUT_PATH = os.path.join(CKPT_DIR, "score_stats.json")

BATCH_SIZE = 64
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_npz(path):
    data = np.load(path, allow_pickle=False)
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_val = data["X_val"]
    Y_val = data["Y_val"]
    mu = data["mu"]
    sigma = data["sigma"]
    return X_train, Y_train, X_val, Y_val, mu, sigma

class WindowDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
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
        out_features = H * in_dim
        self.head = nn.Linear(hidden_ch, out_features)

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

def main():
    X_train, Y_train, X_val, Y_val, mu, sigma = load_npz(NPZ_PATH)

    val_ds = WindowDataset(X_val, Y_val)
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    D = X_val.shape[2]
    H = Y_val.shape[1]

    print("X_val shape:", X_val.shape)
    print("Y_val shape", Y_val.shape)
    print("D =", D)
    print("H =", H)

    model = TCNForecaster(in_dim=D, hidden_ch=64, H=H).to(DEVICE)

    ckpt = torch.load(BEST_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    print(f"Validation loss in checkpoint: {ckpt['val_loss']:.6f}")

    all_feature_mse = []
    all_global_mse = []
    all_vector_l2 = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            xb_tcn = xb.permute(0, 2, 1).contiguous()
            yhat = model(xb_tcn)

            feature_mse, global_mse, vector_l2 = compute_basic_scores(yhat, yb)

            all_feature_mse.append(feature_mse.cpu())
            all_global_mse.append(global_mse.cpu())
            all_vector_l2.append(vector_l2.cpu())
    
    all_feature_mse = torch.cat(all_feature_mse, dim=0)
    all_global_mse = torch.cat(all_global_mse, dim=0)
    all_vector_l2 = torch.cat(all_vector_l2, dim=0)

    eps = 1e-8

    feature_err_mean = all_feature_mse.mean(dim=0)
    feature_err_std = all_feature_mse.std(dim=0)

    z_feature = (all_feature_mse - feature_err_mean) / (feature_err_std + eps)
    z_feature_pos = torch.clamp(z_feature, min=0.0)

    all_std_score = z_feature_pos.mean(dim=1)

    p95 = torch.quantile(all_std_score, 0.95).item()
    p99 = torch.quantile(all_std_score, 0.99).item()

    stats = {
        "num_val_windows": int(all_std_score.shape[0]),
        "std_score_mean": float(all_std_score.mean().item()),
        "std_score_std": float(all_std_score.std().item()),
        "std_score_min": float(all_std_score.min().item()),
        "std_score_max": float(all_std_score.max().item()),
        "std_score_p95": float(p95),
        "std_score_p99": float(p99),
        "recommended_threshold": float(p99),
    }

    print("\nSTD SCORE STATS")
    for k, v in stats.items():
        print(f"{k}: {v}")

        torch.save(
        {
            "feature_mse": all_feature_mse,
            "global_mse": all_global_mse,
            "vector_l2": all_vector_l2,
            "std_score": all_std_score,
            "feature_err_mean": feature_err_mean,
            "feature_err_std": feature_err_std,
        },
        SCORES_OUT_PATH
    )
        
    with open(STATS_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\nSaved scores to:", SCORES_OUT_PATH)
    print("Saved stats to:", STATS_OUT_PATH)

if __name__ == "__main__":
    main()