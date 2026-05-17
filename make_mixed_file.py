from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "clean_data"
NORMAL_DIR = CLEAN_DIR / "normal"
INFECTED_DIR = CLEAN_DIR / "infected"
MIXED_DIR = CLEAN_DIR / "mixed"

NORMAL_FILE = "telemetry_Spotify_clean_1hz.csv"
INFECTED_FILES = [
    "telemetry_browser_YTSAGCYTW_infected_by_sample1_clean_1hz.csv",
    "telemetry_infected_sample_tag_monero_clean_1hz.csv",
    "telemetry_infected_sample_tag_monero_with_browser_act_clean_1hz.csv",
    "telemetry_stock_activity_infected_by_sample1_clean_1hz.csv",
]

TIME_COL = "ts"

NORMAL_BEFORE_ROWS = 2000
INFECTED_ROWS = 2000
NORMAL_AFTER_ROWS = 2000


def take_rows_with_wrap(df: pd.DataFrame, start: int, count: int) -> pd.DataFrame:
    if len(df) == 0:
        raise ValueError("Cannot take rows from an empty dataframe")

    parts = []
    remaining = count
    current = start

    while remaining > 0:
        wrapped_start = current % len(df)
        chunk_size = min(remaining, len(df) - wrapped_start)
        parts.append(df.iloc[wrapped_start:wrapped_start + chunk_size])
        remaining -= chunk_size
        current += chunk_size

    return pd.concat(parts, axis=0, ignore_index=True).copy()


def make_mixed_file(normal_path: Path, infected_path: Path, out_path: Path) -> None:
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

    if len(infected) < INFECTED_ROWS:
        raise ValueError(
            f"{infected_path.name}: need at least {INFECTED_ROWS} infected rows, "
            f"got {len(infected)}"
        )

    normal_before = take_rows_with_wrap(normal, 0, NORMAL_BEFORE_ROWS)
    infected_part = infected.iloc[:INFECTED_ROWS].copy()
    normal_after = take_rows_with_wrap(normal, NORMAL_BEFORE_ROWS, NORMAL_AFTER_ROWS)

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mixed.to_csv(out_path, index=False)

    print("Saved mixed file to:", out_path)
    print("Mixed shape:", mixed.shape)
    print()
    print("Segments:")
    print(f"normal before: rows 0..{NORMAL_BEFORE_ROWS - 1}")
    print(f"infected: rows {NORMAL_BEFORE_ROWS}..{NORMAL_BEFORE_ROWS + INFECTED_ROWS - 1}")
    print(f"normal after: rows {NORMAL_BEFORE_ROWS + INFECTED_ROWS}..{len(mixed) - 1}")
    if len(normal) < NORMAL_BEFORE_ROWS + NORMAL_AFTER_ROWS:
        print(f"normal source wrapped because it has only {len(normal)} rows")
    print()


def main():
    normal_path = NORMAL_DIR / NORMAL_FILE

    print("Normal source:", normal_path)
    print("Creating mixed files in:", MIXED_DIR)
    print()

    for infected_file in INFECTED_FILES:
        infected_path = INFECTED_DIR / infected_file
        sample_name = infected_file.removeprefix("telemetry_").removesuffix("_clean_1hz.csv")
        out_name = f"telemetry_mixed_normal_{sample_name}_normal_clean_1hz.csv"
        out_path = MIXED_DIR / out_name
        make_mixed_file(normal_path, infected_path, out_path)


if __name__ == "__main__":
    main()
