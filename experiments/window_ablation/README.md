# Window Length Ablation Experiment

This experiment checks whether the 120-second input window is justified.

It is isolated from the main project pipeline:

- reads input data from `../../clean_data`
- writes all generated datasets, checkpoints, metrics, tables, and figures into this folder only
- does not modify `../../datasets`, `../../checkpoints`, `../../results`, or the main trained model

## Goal

Train and evaluate the same TCN anomaly detector with different past-window lengths:

```text
T = 30, 60, 120, 180, 240 seconds
```

The forecast horizon stays fixed:

```text
H = 10 seconds
S = 10 seconds
threshold = p99 on validation-normal scores
```

## Run

From the project root:

```powershell
venv\Scripts\python.exe experiments\window_ablation\run_window_ablation.py
```

Quick configuration check without training:

```powershell
venv\Scripts\python.exe experiments\window_ablation\run_window_ablation.py --dry-run
```

Faster smoke run:

```powershell
venv\Scripts\python.exe experiments\window_ablation\run_window_ablation.py --windows 60 120 --epochs 3
```

## Outputs

Per-window outputs:

```text
experiments/window_ablation/runs/T030/
experiments/window_ablation/runs/T060/
experiments/window_ablation/runs/T120/
experiments/window_ablation/runs/T180/
experiments/window_ablation/runs/T240/
```

Aggregate outputs:

- `tables/window_ablation_summary.csv`
- `tables/window_ablation_summary.md`
- `figures/window_ablation_metrics.png`
- `figures/window_ablation_fpr_delay.png`

## Interpretation

For the scientific report, compare:

- mean mixed `F1`
- mean mixed `precision`
- mean mixed `recall`
- mean mixed `false_positive_rate`
- mean detection delay

The selected `T=120` is justified if it gives the best or near-best F1 while avoiding unstable early false events and preserving acceptable false-positive rate.
