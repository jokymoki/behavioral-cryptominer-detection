import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from project_config import DATASET_PATH

PROJECT_DIR = str(PROJECT_ROOT)
NPZ_PATH = str(DATASET_PATH)
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
BEST_PATH = os.path.join(CKPT_DIR, "tcn_best.pt")
LAST_PATH = os.path.join(CKPT_DIR, "tcn_last.pt")

os.makedirs(CKPT_DIR, exist_ok=True)

BATCH_SIZE = 64
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPOCHS = 40
LEARNING_RATE = 1e-3
PATIENCE = 6


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
            dilation = 2 ** i
            blocks.append(TCNBlock(hidden_ch, kernel_size, dilation, dropout))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.blocks(x)
        return x


class TCNForecaster(nn.Module):
    def __init__(self, in_dim, hidden_ch, horizon):
        super().__init__()
        self.backbone = TCNBackbone(in_dim, hidden_ch)
        self.horizon = horizon
        self.head = nn.Linear(hidden_ch, horizon * in_dim)

    def forward(self, x):
        z = self.backbone(x)
        z_last = z[:, :, -1]
        y_flat = self.head(z_last)
        batch_size = x.shape[0]
        y = y_flat.view(batch_size, self.horizon, -1)
        return y


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            xb_tcn = xb.permute(0, 2, 1).contiguous()
            yhat = model(xb_tcn)

            loss = criterion(yhat, yb)
            running_loss += loss.item()

    return running_loss / max(1, len(loader))


def main():
    X_train, Y_train, X_val, Y_val, mu, sigma = load_npz(NPZ_PATH)

    mu_t = torch.from_numpy(mu.astype(np.float32))
    sigma_t = torch.from_numpy(sigma.astype(np.float32))

    train_ds = WindowDataset(X_train, Y_train)
    val_ds = WindowDataset(X_val, Y_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("Train shape:", X_train.shape, Y_train.shape)
    print("Val shape:", X_val.shape, Y_val.shape)

    xb, yb = next(iter(train_loader))
    print("One batch X/Y:", xb.shape, yb.shape)

    xb = xb.to(DEVICE)
    yb = yb.to(DEVICE)

    xb_tcn = xb.permute(0, 2, 1).contiguous()
    print("xb_tcn:", xb_tcn.shape)

    _, T, D = xb.shape
    H = yb.shape[1]
    print(f"T = {T}, H = {H}, D = {D}")

    model = TCNForecaster(in_dim=D, hidden_ch=64, horizon=H).to(DEVICE)

    yhat = model(xb_tcn)
    print("yhat shape:", yhat.shape)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val = float("inf")
    last_val = None
    epochs_no_improve = 0

    train_losses = []
    val_losses = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            xb_tcn = xb.permute(0, 2, 1).contiguous()
            yhat = model(xb_tcn)

            loss = criterion(yhat, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"epoch {epoch} batch {batch_idx}: loss={loss.item():.6f}")

        avg_train_loss = running_loss / max(1, len(train_loader))
        avg_val_loss = evaluate(model, val_loader, criterion)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(
            f"epoch {epoch} done: "
            f"avg_train_loss={avg_train_loss:.6f}, "
            f"avg_val_loss={avg_val_loss:.6f}"
        )

        last_val = avg_val_loss

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "mu": mu_t,
                "sigma": sigma_t,
                "train_losses": train_losses,
                "val_losses": val_losses,
            },
            LAST_PATH,
        )

        if avg_val_loss < best_val:
            best_val = avg_val_loss
            epochs_no_improve = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "mu": mu_t,
                    "sigma": sigma_t,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                },
                BEST_PATH,
            )

            print(f"new best model saved with val_loss={avg_val_loss:.6f}")
        else:
            epochs_no_improve += 1
            print(f"no improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= PATIENCE:
            print(f"early stopping at epoch {epoch}")
            break

    print(f"training done: best_val={best_val:.6f}, last_val={last_val:.6f}")
    print(f"checkpoints: best={BEST_PATH}, last={LAST_PATH}")

    ckpt = torch.load(BEST_PATH, map_location=DEVICE, weights_only=False)
    print(
        f"best checkpoint summary: "
        f"epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}"
    )


if __name__ == "__main__":
    main()
