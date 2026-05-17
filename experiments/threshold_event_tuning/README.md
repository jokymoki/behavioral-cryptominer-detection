# Threshold and Event Rule Tuning

This experiment tunes detection threshold and minimum event length without retraining.

It is isolated:

- reads the current validation scores from `../../checkpoints/val_scores_from_script.pt`
- reads current mixed score files from `../../checkpoints/mixed_dataset`
- writes outputs only into `experiments/threshold_event_tuning`
- does not modify datasets, checkpoints, or main results

## Run

From project root:

```powershell
venv\Scripts\python.exe experiments\threshold_event_tuning\run_threshold_event_tuning.py
```

## Outputs

- `tables/threshold_event_tuning.csv`
- `tables/threshold_event_tuning.md`
- `figures/threshold_event_tuning_f1_fpr.png`

## Interpretation

Prefer configurations with:

- high F1
- recall >= 0.98
- lower false positive rate
- reasonable detection delay
