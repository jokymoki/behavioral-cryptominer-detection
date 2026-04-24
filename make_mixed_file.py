from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "clean_data"

NORMAL_FILE = "telemetry_Spotify_clean_1hz.csv"
INFECTED_FILE = "telemetry_second_sample_of_virus_clean_1hz.csv"

OUT_FILE = "telemetry_mixed_normal_infected_normal_clean_1hz.csv"

TIME_COL = "ts"

NORMAL_BEFORE_ROWS = 2000
INFECTED_ROWS = 2000
NORMAL_AFTER_ROWS = 2000


def main():
    normal_path = CLEAN_DIR / NORMAL_FILE
    infected_path = CLEAN_DIR / INFECTED_FILE

    normal = pd.read_csv(normal_path)
    infected = pd.read_csv(infected_path)

    if TIME_COL not in normal.columns:
        raise ValueError(f"Missing {TIME_COL} in normal file")

    if TIME_COL not in infected.columns:
        raise ValueError(f"Missing {TIME_COL} in infected file")

    normal_cols = list(normal.columns)
    infected_cols = list(infected.columns)

    if normal_cols != infected_cols:
        raise ValueError("Column mismatch between normal and infected files")

    normal_before = normal.iloc[:NORMAL_BEFORE_ROWS].copy()
    infected_part = infected.iloc[:INFECTED_ROWS].copy()
    normal_after = normal.iloc[NORMAL_BEFORE_ROWS:NORMAL_BEFORE_ROWS + NORMAL_AFTER_ROWS].copy()

    mixed = pd.concat(
        [normal_before, infected_part, normal_after],
        axis=0,
        ignore_index=True
    )

    # Rebuild timestamp column as artificial 1 Hz timeline
    start_time = pd.to_datetime(normal[TIME_COL].iloc[0])
    mixed[TIME_COL] = pd.date_range(
        start=start_time,
        periods=len(mixed),
        freq="1s"
    )

    out_path = CLEAN_DIR / OUT_FILE
    mixed.to_csv(out_path, index=False)

    print("Saved mixed file to:", out_path)
    print("Mixed shape:", mixed.shape)
    print()
    print("Segments:")
    print(f"normal before: rows 0..{NORMAL_BEFORE_ROWS - 1}")
    print(f"infected: rows {NORMAL_BEFORE_ROWS}..{NORMAL_BEFORE_ROWS + INFECTED_ROWS - 1}")
    print(f"normal after: rows {NORMAL_BEFORE_ROWS + INFECTED_ROWS}..{len(mixed) - 1}")


if __name__ == "__main__":
    main()