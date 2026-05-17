# Classical Baselines Experiment

This experiment compares the DL TCN anomaly detector with classical one-class methods:

- Isolation Forest
- One-Class SVM

It is isolated from the main project pipeline:

- reads input data from `../../clean_data`
- reads the main DL report from `../../results/tables/summary_metrics.csv` when available
- writes all generated artifacts into `experiments/classical_baselines`
- does not modify `../../datasets`, `../../checkpoints`, `../../results`, or the main trained model

## Method

The classical models use the same main time configuration:

```text
T = 30 seconds
S = 10 seconds
```

For every 30-second telemetry window, a fixed feature vector is built from each raw telemetry feature:

```text
mean, std, min, max, last - first
```

The models are trained only on normal windows from `clean_data/normal`.

The anomaly threshold is selected as the 99th percentile of validation-normal anomaly scores, matching the DL evaluation logic.

## Run

From the project root:

```powershell
venv\Scripts\python.exe experiments\classical_baselines\run_classical_baselines.py
```

## Outputs

```text
experiments/classical_baselines/results/
experiments/classical_baselines/tables/
experiments/classical_baselines/figures/
experiments/classical_baselines/models/
```

Key files:

- `tables/classical_baselines_summary.csv`
- `tables/classical_baselines_summary.md`
- `tables/model_comparison_with_dl.csv`
- `tables/model_comparison_with_dl.md`
- `figures/model_comparison_f1_fpr.png`
- `figures/model_comparison_precision_recall.png`

## Interpretation

For the scientific report, compare:

- precision
- recall
- F1-score
- false positive rate
- detection delay

The DL model is stronger if it achieves higher F1/recall at comparable or lower false-positive rate and avoids unstable early false events.
