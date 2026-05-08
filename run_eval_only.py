import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "score_windows.py",
    "score_infected_file.py",
    "evaluate_experiments.py",
    "plot_mixed_scores.py",
    "plot_normal_vs_infected.py",
    "plot_infected_scores.py",
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
    print("Starting evaluation-only pipeline")
    print("This pipeline does NOT rebuild dataset and does NOT train the model.")

    for script in SCRIPTS:
        run_script(script)

    print()
    print("=" * 80)
    print("Evaluation-only pipeline completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()