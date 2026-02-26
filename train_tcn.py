import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
#imports for NN
import torch.nn as nn
import torch.nn.functional as F

NPZ_PATH = r"C:\Users\jokym\Desktop\Project_DL\datasets\windows_T120_H10_S10.npz"
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
class CasualConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super.__init__()
        self.pad = (kernel_size - 1) *dilation #!!!WHATCHOUT TOMOROW
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
    
    def forward(self, x):
        #x: (B, C, T)
        x = F.pad(x, (self.pad, 0)) # pad only on left
        return self.conv(x)
    
#residual-block
class TCNBlock(nn.Module):
    def __inti__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = CasualConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CasualConv1d(channels, channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
    
    def forwar(self, x):
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
            d = 2**i #1,2,3,4,8,16,32...
            blocks.append(TCNBlock(hidden_ch, kernel_size, d, dropout))
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        #x: (B, D, T)
        x = self.in_proj(x)  #(B, C, T)
        x = self.blocks(x)   #(B, C, T)
        return x

def main():
    #making dataloader
    X_train, Y_train, X_val, Y_val, mu, sigma = load_npz(NPZ_PATH)

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
    

if __name__ == "__main__":
    main()