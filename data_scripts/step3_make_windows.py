from pathlib import Path
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from project_config import H, NORMAL_HOLDOUT_FILES, S, T, get_feature_columns

BASE_DIR = PROJECT_ROOT
CLEAN_DIR = BASE_DIR / "clean_data" / "normal"


def make_windows(X: np.ndarray, T: int, H: int, S: int):
    """
    X: (N, D)
    returns:
      X_past:   (M, T, D)
      Y_future: (M, H, D)
    """
    N, D = X.shape
    M = (N - (T + H)) // S + 1

    if M <= 0:
        empty_X = np.zeros((0, T, D), dtype=np.float32)
        empty_Y = np.zeros((0, H, D), dtype=np.float32)
        return empty_X, empty_Y

    X_past = np.zeros((M, T, D), dtype=np.float32)
    Y_future = np.zeros((M, H, D), dtype=np.float32)

    i = 0
    for start in range(0, N - (T + H) + 1, S):
        X_past[i] = X[start: start + T]
        Y_future[i] = X[start + T: start + T + H]
        i += 1

    return X_past, Y_future


def main():
    files = [
        f for f in sorted(CLEAN_DIR.glob("*.csv"))
        if f.name not in NORMAL_HOLDOUT_FILES
    ]
    print("Found files:", len(files))
    if NORMAL_HOLDOUT_FILES:
        print("Holdout normal files excluded from training:", ", ".join(NORMAL_HOLDOUT_FILES))
    if not files:
        return

    # ---- Single file test (first file) ----
    f0 = files[0]
    df0 = pd.read_csv(f0)
    feature_columns = get_feature_columns(df0.columns)
    X0 = df0[feature_columns].to_numpy(dtype=np.float32)

    print("File:", f0.name)
    print("Feature columns:", len(feature_columns))
    print("X shape (N, D):", X0.shape)

    N0, D0 = X0.shape
    M0 = (N0 - (T + H)) // S + 1
    print(f"N: {N0}, D: {D0}, M: {M0}")

    X_past0, Y_future0 = make_windows(X0, T, H, S)
    print("X_past shape:", X_past0.shape)
    print("Y_future shape:", Y_future0.shape)

    # sanity checks (should match)
    print("check past last:", X_past0[0, T - 1, 0], "==", X0[T - 1, 0])
    print("check future first:", Y_future0[0, 0, 0], "==", X0[T, 0])

    # ---- All files -> build dataset ----
    Xp_list = []
    Yf_list = []

    for f in files:
        df = pd.read_csv(f)
        current_feature_columns = get_feature_columns(df.columns)
        if current_feature_columns != feature_columns:
            raise ValueError(f"{f.name}: feature columns do not match reference file")
        X = df[feature_columns].to_numpy(dtype=np.float32)

        X_past, Y_future = make_windows(X, T, H, S)
        if X_past.shape[0] > 0:
            Xp_list.append(X_past)
            Yf_list.append(Y_future)

    X_all = np.concatenate(Xp_list, axis=0)
    Y_all = np.concatenate(Yf_list, axis=0)

    print("FINAL DATASET:")
    print("X_all shape:", X_all.shape)
    print("Y_all shape:", Y_all.shape)

    # ---- Step A: train/val split ----
    n = X_all.shape[0]
    train_frac = 0.8
    n_train = int(n * train_frac)

    X_train = X_all[:n_train]
    Y_train = Y_all[:n_train]
    X_val = X_all[n_train:]
    Y_val = Y_all[n_train:]

    print("Train shape:", X_train.shape)
    print("Val shape:", X_val.shape)

    # ---- Step B: normalization (fit on train only) ----
    mu = X_train.mean(axis=(0, 1))           # (D,)
    sigma = X_train.std(axis=(0, 1))         # (D,)

    # IMPORTANT: protect from zero/tiny std
    sigma = np.where(sigma < 1e-8, 1.0, sigma)

    print("mu shape:", mu.shape)
    print("sigma shape:", sigma.shape)
    print("min sigma:", float(sigma.min()))

    # Apply normalization to BOTH X and Y using the SAME mu/sigma
    X_train = (X_train - mu) / sigma
    X_val = (X_val - mu) / sigma
    Y_train = (Y_train - mu) / sigma
    Y_val = (Y_val - mu) / sigma

    # quick sanity check: train should be ~mean 0, std ~1
    m_check = X_train.mean(axis=(0, 1))[:5]
    s_check = X_train.std(axis=(0, 1))[:5]
    print("X_train mean first 5 feats:", m_check)
    print("X_train std  first 5 feats:", s_check)

    # ---- Step C: save ----
    OUT_DIR = BASE_DIR / "datasets"
    OUT_DIR.mkdir(exist_ok=True)

    out_path = OUT_DIR / f"windows_T{T}_H{H}_S{S}.npz"

    np.savez_compressed(
        out_path,
        X_train=X_train.astype(np.float32),
        Y_train=Y_train.astype(np.float32),
        X_val=X_val.astype(np.float32),
        Y_val=Y_val.astype(np.float32),
        mu=mu.astype(np.float32),
        sigma=sigma.astype(np.float32),
        feature_columns=np.array(feature_columns),
    )

    print("Saved dataset to:", out_path.resolve())


if __name__ == "__main__":
    main()
