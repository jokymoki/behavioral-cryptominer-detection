import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
#imports for NN
import torch.nn as nn
import torch.nn.functional as F

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(PROJECT_DIR, "datasets", "windows_T120_H10_S10.npz")
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
BEST_PATH = os.path.join(CKPT_DIR, "tcn_best.pt")
LAST_PATH = os.path.join(CKPT_DIR, "tcn_last.pt")
SCORES_PATH = os.path.join(CKPT_DIR, "val_scores.pt")
os.makedirs(CKPT_DIR, exist_ok=True)
BATCH_SIZE = 64
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#function for loading in arrays from NPZ file format
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
    #initializing 
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))
    #telling the length from (N, T, D) it is N so [0]
    def __len__(self):
        return self.X.shape[0]
    
    #telling i
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    

#1D convert
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) *dilation #!!!WHATCHOUT TOMOROW
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
    
    def forward(self, x):
        #x: (B, C, T)
        x = F.pad(x, (self.pad, 0)) # pad only on left
        return self.conv(x)
    
#residual-block
class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        #x: (B, C, T)
        out = self.conv1(x)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = F.relu(out)
        out = self.dropout(out)

        return x + out #residual


#backbone
class TCNBackbone(nn.Module):
    def __init__(self, in_dim, hidden_ch=64, kernel_size=3, levels=6, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Conv1d(in_dim, hidden_ch, kernel_size=1)

        blocks = []
        for i in range(levels):
            d = 2**i # 1,2,4,8,16,32...
            blocks.append(TCNBlock(hidden_ch, kernel_size, d, dropout))
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        #x: (B, D, T)
        x = self.in_proj(x)  #(B, C, T)
        x = self.blocks(x)   #(B, C, T)
        return x

#TCN Forecasting module
class TCNForecaster(nn.Module):
    def __init__(self, in_dim, hidden_ch, H):
        super().__init__()
        self.backbone = TCNBackbone(in_dim, hidden_ch)
        self.H = H
        out_features = H * in_dim
        self.head = nn.Linear(hidden_ch, out_features)
    def forward(self, x):
        #x: (B, D, T):
        z = self.backbone(x) #(B, C, T)
        z_last = z[:,:,-1]   #(B, C) последний timestamp
        y_flat = self.head(z_last) #(B, H*D)
        B = x.shape[0]
        y = y_flat.view(B, self.H, -1) #(B, H, D)
        return y



def main():
    #making dataloader
    X_train, Y_train, X_val, Y_val, mu, sigma = load_npz(NPZ_PATH)
    mu_t = torch.from_numpy(mu.astype(np.float32))
    sigma_t = torch.from_numpy(sigma.astype(np.float32))

    train_ds = WindowDataset(X_train, Y_train)
    val_ds = WindowDataset(X_val, Y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers = NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size = BATCH_SIZE, shuffle=False, num_workers = NUM_WORKERS)


    #one batch test
    xb, yb = next(iter(train_loader))
    print(xb.shape, yb.shape)
    xb = xb.to(DEVICE)
    yb = yb.to(DEVICE)

    #for Conv1D/TCN we usually want (B, D, T)
    xb_tcn = xb.permute(0, 2, 1).contiguous()
    print("xb_tcn:", xb_tcn.shape)

    #one more check of target
    B, T, D = xb.shape
    H = yb.shape[1]
    print(f"T = {T}, H = {H}, D = {D}")

    #temporary checking
    model = TCNForecaster(in_dim=D, hidden_ch=64, H=H).to(DEVICE)
    yhat = model(xb_tcn)
    print("yhat shape:", yhat.shape)

    #error counter
    criterion = nn.MSELoss()

    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #training loop
    best_val = float("inf")
    last_val = None
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (xb, yb) in enumerate(train_loader):
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            # for Conv1D/TCN we want (B, D, T)
            xb_tcn = xb.permute(0, 2, 1).contiguous()

            yhat = model(xb_tcn)
            loss = criterion(yhat, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 50 == 0:
                print(f"epoch {epoch} batch {batch_idx}: loss={loss.item():.6f}")

        avg_loss = running_loss / max(1, len(train_loader))
        print(f"epoch {epoch} done: avg_train_loss={avg_loss:.6f}")

        #validation loop
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)

                xb_tcn = xb.permute(0, 2, 1).contiguous()
                yhat = model(xb_tcn)
                vloss = criterion(yhat, yb)

                val_running_loss += vloss.item()

        avg_val_loss = val_running_loss / max(1, len(val_loader))
        print(f"epoch {epoch} done: avg_val_loss={avg_val_loss:.6f}")
        last_val = avg_val_loss

        # save last checkpoint every epoch
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "mu": mu_t,
            "sigma": sigma_t,
        }, LAST_PATH)

        # save best checkpoint
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": avg_val_loss,
                "mu": mu_t,
                "sigma": sigma_t,
            }, BEST_PATH)
            print(f"new best model saved with val_loss={avg_val_loss:.6f}")

    print(f"training done: best_val={best_val:.6f}, last_val={last_val:.6f}")
    print(f"checkpoints: best={BEST_PATH}, last={LAST_PATH}")

    # load BEST checkpoint for inference/next steps
    ckpt = torch.load(BEST_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded BEST checkpoint from epoch {ckpt['epoch']} with val_loss={ckpt['val_loss']:.6f}")


    
    def compute_basic_scores(pred, target):
        err = pred - target
        sq_err = err**2

        feature_mse = sq_err.mean(dim=1) #(B, D)
        global_mse = feature_mse.mean(dim=1) #(B)
        vector_l2 = torch.sqrt((feature_mse ** 2).sum(dim=1)) #(B)

        return feature_mse, global_mse, vector_l2


    
    
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

    feature_err_mean = all_feature_mse.mean(dim=0)   # (D,)
    feature_err_std = all_feature_mse.std(dim=0)     # (D,)

    z_feature = (all_feature_mse - feature_err_mean) / (feature_err_std + eps)
    z_feature_pos = torch.clamp(z_feature, min=0.0)

    all_std_score = z_feature_pos.mean(dim=1)        # (N_val,)

    print("all_feature_mse shape:", all_feature_mse.shape)
    print("all_global_mse shape:", all_global_mse.shape)
    print("all_vector_l2 shape:", all_vector_l2.shape)

    print("\nGLOBAL MSE STATS")
    print("mean:", all_global_mse.mean().item())
    print("std:", all_global_mse.std().item())
    print("min:", all_global_mse.min().item())
    print("max:", all_global_mse.max().item())
    print("p90:", torch.quantile(all_global_mse, 0.90).item())
    print("p95:", torch.quantile(all_global_mse, 0.95).item())
    print("p99:", torch.quantile(all_global_mse, 0.99).item())

    print("\nVECTOR L2 STATS")
    print("mean:", all_vector_l2.mean().item())
    print("std:", all_vector_l2.std().item())
    print("min:", all_vector_l2.min().item())
    print("max:", all_vector_l2.max().item())
    print("p90:", torch.quantile(all_vector_l2, 0.90).item())
    print("p95:", torch.quantile(all_vector_l2, 0.95).item())
    print("p99:", torch.quantile(all_vector_l2, 0.99).item())

    print("\nSTANDARDIZED POSITIVE SCORE STATS")
    print("mean:", all_std_score.mean().item())
    print("std:", all_std_score.std().item())
    print("min:", all_std_score.min().item())
    print("max:", all_std_score.max().item())
    print("p90:", torch.quantile(all_std_score, 0.90).item())
    print("p95:", torch.quantile(all_std_score, 0.95).item())
    print("p99:", torch.quantile(all_std_score, 0.99).item())

    print("\nfeature_err_mean shape:", feature_err_mean.shape)
    print("feature_err_std shape:", feature_err_std.shape)
    print("min feature_err_std:", feature_err_std.min().item())
    print("max feature_err_std:", feature_err_std.max().item())

    torch.save({
        "feature_mse": all_feature_mse,
        "global_mse": all_global_mse,
        "vector_l2": all_vector_l2,
        "std_score": all_std_score,
        "feature_err_mean": feature_err_mean,
        "feature_err_std": feature_err_std,
    }, SCORES_PATH)

    print("\nsaved scores to val_scores.pt")
if __name__ == "__main__":
    main()