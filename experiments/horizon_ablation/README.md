# Horizon Ablation Experiment

This experiment checks whether the 10-second forecast horizon is justified.

It is isolated from the main project pipeline:

- reads input data from `../../clean_data`
- writes all generated datasets, checkpoints, metrics, tables, and figures into this folder only
- does not modify `../../datasets`, `../../checkpoints`, `../../results`, or the main trained model

## Goal

Train and evaluate the same TCN anomaly detector with different forecast horizons:

```text
H = 1, 5, 10, 20, 30 seconds
```

All other major parameters stay fixed:

```text
T = 120 seconds
S = 10 seconds
normal train data = clean_data/normal
infected test data = clean_data/infected
mixed test data = clean_data/mixed
threshold = p99 on validation-normal scores
```

## Run

From the project root:

```powershell
venv\Scripts\python.exe experiments\horizon_ablation\run_horizon_ablation.py
```

Quick configuration check without training:

```powershell
venv\Scripts\python.exe experiments\horizon_ablation\run_horizon_ablation.py --dry-run
```

Faster smoke run:

```powershell
venv\Scripts\python.exe experiments\horizon_ablation\run_horizon_ablation.py --horizons 5 10 --epochs 3
```

## Outputs

Per-horizon outputs:

```text
experiments/horizon_ablation/runs/H001/
experiments/horizon_ablation/runs/H005/
experiments/horizon_ablation/runs/H010/
experiments/horizon_ablation/runs/H020/
experiments/horizon_ablation/runs/H030/
```

Each run contains:

- `dataset/windows.npz`
- `checkpoints/tcn_best.pt`
- `checkpoints/val_scores.pt`
- `checkpoints/score_stats.json`
- `results/infected_evaluation.json`
- `results/mixed_evaluation.json`

Aggregate outputs:

- `tables/horizon_ablation_summary.csv`
- `tables/horizon_ablation_summary.md`
- `figures/horizon_ablation_metrics.png`
- `figures/horizon_ablation_fpr_delay.png`

## Interpretation

For the scientific report, compare:

- mean mixed `F1`
- mean mixed `precision`
- mean mixed `recall`
- mean mixed `false_positive_rate`
- mean detection delay

The selected `H=10` is justified if it gives the best or near-best F1 while keeping FPR and detection delay acceptable compared with shorter and longer horizons.
