from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

TIME_COL = "ts"

# Main model configuration selected after ablation experiments.
T = 30
H = 10
S = 10

DATASET_PATH = BASE_DIR / "datasets" / f"windows_T{T}_H{H}_S{S}.npz"

TRUE_INFECTED_START_SEC = 2000
TRUE_INFECTED_END_SEC = 4000
MIN_CONSECUTIVE_WINDOWS = 4
SCORE_KEY = "std_score_top5"

# Normal files reserved for test/mixed backgrounds. They must not be used for
# fitting the normal baseline or validation threshold.
NORMAL_HOLDOUT_FILES = [
    "telemetry_Spotify_clean_1hz.csv",
]
