import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "step2_check_columns.py",
    "step3_make_windows.py",
    "train_tcn.py",
    "score_windows.py",
]


def run_script(script_name):
    script_path = BASE_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")
    
    print()
    print("=" * 80)
    print(f"Running: {script_name}")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {script_name}")
    
def main():
    print("Starting SAFE training pipeline")
    print("This pipeline uses only clean_data/normal.")
    print("It does NOT use infected or mixed files.")

    for script in SCRIPTS:
        run_script(script)

    print()
    print("=" * 80)
    print("Safe training pipeline completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()