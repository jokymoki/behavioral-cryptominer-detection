import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "score_windows.py",
    "score_infected_dataset.py",
    "score_mixed_dataset.py",
    "make_report_artifacts.py",
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
    print("Starting evaluation-only pipeline", flush=True)
    print("This pipeline does NOT rebuild dataset and does NOT train the model.", flush=True)
    print("It prints validation, infected-only, and mixed-scenario results.", flush=True)
    print("It also saves report tables and figures.", flush=True)

    for script in SCRIPTS:
        run_script(script)

    print()
    print("=" * 80)
    print("Evaluation-only pipeline completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
