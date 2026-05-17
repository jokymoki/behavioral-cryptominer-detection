import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "data_scripts/step2_check_columns.py",
    "data_scripts/step3_make_windows.py",
    "model_scripts/train_tcn.py",
    "model_scripts/score_windows.py",
    "data_scripts/make_mixed_file.py",
    "model_scripts/score_infected_dataset.py",
    "model_scripts/score_mixed_dataset.py",
    "report_scripts/make_report_artifacts.py",
]


def run_script(script_name):
    script_path = BASE_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")
    
    print()
    print("=" * 80)
    print(f"Running: {script_name}")
    print("=" * 80, flush=True)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {script_name}")
    
def main():
    print("Starting full SAFE training pipeline", flush=True)
    print("Training uses only clean_data/normal.", flush=True)
    print("Infected and mixed files are used only after training for evaluation/reporting.", flush=True)

    for script in SCRIPTS:
        run_script(script)

    print()
    print("=" * 80)
    print("Full safe training pipeline completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
