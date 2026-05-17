import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# ---------------- CONFIG ----------------
# Put your raw CSV(s) in RAW_DIR. Cleaned files will go to OUT_DIR.
RAW_DIR = BASE_DIR / "raw_data"
OUT_DIR = BASE_DIR / "clean_data"

TIME_COL = "ts"
DROP_COLS = ["scenario"]  # non-numeric label/metadata columns to drop
FREQ = pd.Timedelta(seconds=1)  # robust across pandas versions
# ---------------------------------------


def clean_one_file(in_path: Path, out_path: Path) -> None:
    df = pd.read_csv(in_path)

    if TIME_COL not in df.columns:
        raise ValueError(f"{in_path.name}: missing required time column '{TIME_COL}'")

    # 1) Parse timestamps + sort
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL]).sort_values(TIME_COL).reset_index(drop=True)

    # remove duplicate timestamps
    df = df.drop_duplicates(subset=[TIME_COL], keep="last")

    # 2) Drop non-feature columns if present
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    # 3) Convert to numeric (except time), handle inf/NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    feature_cols = [c for c in df.columns if c != TIME_COL]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    nans_before = int(df[feature_cols].isna().sum().sum())

    # 4) Reindex to exact 1 Hz grid
    df = df.set_index(TIME_COL)

    if df.index.min() is pd.NaT or df.index.max() is pd.NaT:
        raise ValueError(f"{in_path.name}: invalid timestamps after parsing")

    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=FREQ)
    df = df.reindex(full_index)

    # 5) Fill missing seconds
    df = df.ffill().fillna(0.0)

    # 6) Sanity check: ensure 1-second steps
    diffs = pd.Series(df.index).diff().dt.total_seconds().dropna()
    bad_steps = int((diffs != 1.0).sum())

    nans_after = int(df.isna().sum().sum())

    # 7) Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = df.reset_index().rename(columns={"index": TIME_COL})
    df_out.to_csv(out_path, index=False)

    print("[OK]", in_path.name)
    print(f"   rows_out: {len(df_out)} | cols: {len(df_out.columns)}")
    print(f"   NaNs before: {nans_before} | NaNs after: {nans_after}")
    print(f"   Non-1s steps after: {bad_steps} (should be 0)")
    print()


def main():
    RAW_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        print(f"[ERROR] No CSV files found in: {RAW_DIR.resolve()}")
        print("Put your raw telemetry CSVs into raw_data/ and run again.")
        return

    print(f"Found {len(files)} file(s) in {RAW_DIR.resolve()}")
    print(f"Saving cleaned files to {OUT_DIR.resolve()}\n")

    for in_path in files:
        out_path = OUT_DIR / (in_path.stem + "_clean_1hz.csv")
        try:
            clean_one_file(in_path, out_path)
        except Exception as e:
            print("[ERROR] ERROR on", in_path.name)
            print("   ", repr(e))
            print()

    print("DONE.")


if __name__ == "__main__":
    main()
