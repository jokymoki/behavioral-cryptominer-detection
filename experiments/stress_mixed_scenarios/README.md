# Stress Mixed Scenarios

This folder contains synthetic stress tests for model comparison.

The goal is to build unusual mixed scenarios that may be harder for classical one-class models than the regular `clean_data/mixed` files.

The experiment is isolated:

- reads `../../clean_data`
- reads the main DL checkpoint from `../../checkpoints`
- reads classical models from `../classical_baselines/models`
- writes generated mixed CSVs and results only into this folder
- does not modify project `clean_data`, `datasets`, `checkpoints`, or `results`

## Scenario

`stealth_blend`:

- normal before segment from Spotify
- infected segment is a blend between normal background and infected telemetry
- normal after segment from Spotify

Blend formula:

```text
stealth = normal_background + alpha * (infected - normal_background)
```

Low `alpha` means the infected segment is more hidden.

## Run

From project root:

```powershell
venv\Scripts\python.exe experiments\stress_mixed_scenarios\run_stress_mixed.py
```

Outputs:

- `generated/*.csv`
- `results/stress_mixed_results.json`
- `tables/stress_mixed_summary.csv`
- `tables/stress_mixed_summary.md`
- `figures/stress_mixed_f1_recall.png`
