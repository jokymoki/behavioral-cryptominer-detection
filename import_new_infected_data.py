from pathlib import Path
import shutil

from fixing_data import clean_one_file


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = Path(r"C:\Users\jokym\Downloads\expirement\expirement")
RAW_DIR = BASE_DIR / "raw_data"
INFECTED_CLEAN_DIR = BASE_DIR / "clean_data" / "infected"

NEW_INFECTED_FILES = [
    "telemetry_browser_YTSAGCYTW_infected_by_sample1.csv",
    "telemetry_infected_sample_tag_monero.csv",
    "telemetry_infected_sample_tag_monero_with_browser_act.csv",
    "telemetry_stock_activity_infected_by_sample1.csv",
]


def import_one(source_path: Path) -> Path:
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source file: {source_path}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / source_path.name
    shutil.copy2(source_path, raw_path)
    return raw_path


def main() -> None:
    INFECTED_CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    print("Importing infected telemetry files from:")
    print(DEFAULT_SOURCE_DIR)
    print()

    for file_name in NEW_INFECTED_FILES:
        source_path = DEFAULT_SOURCE_DIR / file_name
        raw_path = import_one(source_path)
        clean_path = INFECTED_CLEAN_DIR / f"{raw_path.stem}_clean_1hz.csv"

        print(f"Cleaning infected sample: {raw_path.name}")
        clean_one_file(raw_path, clean_path)

    print("Imported and cleaned infected files:")
    for file_name in NEW_INFECTED_FILES:
        print(" -", INFECTED_CLEAN_DIR / f"{Path(file_name).stem}_clean_1hz.csv")


if __name__ == "__main__":
    main()
