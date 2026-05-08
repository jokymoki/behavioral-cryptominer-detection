import os
import json
import torch
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")

RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(RESULTS_DIR, "mixed_evaluation.json")

SCORES_PATH = os.path.join(CKPT_DIR, "mixed_scores.pt")
STATS_PATH = os.path.join(CKPT_DIR, "score_stats.json")

SCORE_KEY = "std_score_top5"

T = 120
H = 10
STRIDE = 10
MIN_CONSECUTIVE = 3

TRUE_INFECTED_START_SEC = 2000
TRUE_INFECTED_END_SEC = 4000


def find_consecutive_runs(flags):
    runs = []
    start = None

    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None

    if start is not None:
        runs.append((start, len(flags) - 1))

    return runs


def filter_runs_by_length(runs, min_length):
    result = []

    for start, end in runs:
        length = end - start + 1
        if length >= min_length:
            result.append((start, end, length))

    return result


def safe_div(a, b):
    if b == 0:
        return 0.0
    return a / b


def build_true_labels(num_windows):
    true_positive = np.zeros(num_windows, dtype=bool)

    for i in range(num_windows):
        window_start = i * STRIDE
        window_end = window_start + T + H

        overlaps_infected = (
            window_start < TRUE_INFECTED_END_SEC
            and window_end > TRUE_INFECTED_START_SEC
        )

        true_positive[i] = overlaps_infected

    return true_positive


def main():
    data = torch.load(SCORES_PATH, map_location="cpu", weights_only=False)

    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)

    score = data[SCORE_KEY]
    if isinstance(score, torch.Tensor):
        score = score.cpu().numpy()

    threshold = stats["recommended_threshold"]

    predicted_positive = score > threshold
    true_positive = build_true_labels(len(predicted_positive))

    tp = int(np.logical_and(predicted_positive, true_positive).sum())
    fp = int(np.logical_and(predicted_positive, ~true_positive).sum())
    fn = int(np.logical_and(~predicted_positive, true_positive).sum())
    tn = int(np.logical_and(~predicted_positive, ~true_positive).sum())

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    false_positive_rate = safe_div(fp, fp + tn)

    runs = find_consecutive_runs(predicted_positive)
    events = filter_runs_by_length(runs, MIN_CONSECUTIVE)

    print("=== Mixed scenario evaluation ===")
    print("Score key:", SCORE_KEY)
    print("Threshold:", threshold)
    print("Total windows:", len(score))
    print("Window length T:", T)
    print("Forecast horizon H:", H)
    print("Stride:", STRIDE)
    print()

    print("=== True infected interval ===")
    print(f"True infected interval: {TRUE_INFECTED_START_SEC}s..{TRUE_INFECTED_END_SEC}s")
    print("Label rule: window is infected if [window_start, window_start + T + H] overlaps true infected interval")
    print()

    print("=== Confusion matrix by windows ===")
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print("TN:", tn)
    print()

    print("=== Metrics ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"FPR:       {false_positive_rate:.4f}")
    print()

    print("=== Detected events ===")

    if not events:
        print(f"No events with at least {MIN_CONSECUTIVE} consecutive windows.")
        return

    first_event_start_sec = None

    for idx, (start_w, end_w, length) in enumerate(events, start=1):
        start_sec = start_w * STRIDE
        end_sec = end_w * STRIDE

        if first_event_start_sec is None:
            first_event_start_sec = start_sec

        print(
            f"event {idx}: "
            f"windows {start_w}..{end_w}, "
            f"length={length}, "
            f"time={start_sec}s..{end_sec}s"
        )

    print()

    detection_delay = first_event_start_sec - TRUE_INFECTED_START_SEC

    print("=== Detection timing ===")
    print("True infected start:", TRUE_INFECTED_START_SEC, "s")
    print("First detected event start:", first_event_start_sec, "s")
    print("Detection delay:", detection_delay, "s")


    #results
    results = {
    "score_key": SCORE_KEY,
    "threshold": float(threshold),

    "window_config": {
            "T": T,
            "H": H,
            "stride": STRIDE,
    },

    "true_infected_interval_sec": {
            "start": TRUE_INFECTED_START_SEC,
            "end": TRUE_INFECTED_END_SEC,
    },

    "confusion_matrix": {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
    },

    "metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "false_positive_rate": float(false_positive_rate),
    },

    "detection": {
            "first_detected_event_start_sec": first_event_start_sec,
            "detection_delay_sec": detection_delay,
    },

    "events": [
            {
                "event_id": idx,
                "start_window": start_w,
                "end_window": end_w,
                "length_windows": length,
                "start_sec": start_w * STRIDE,
                "end_sec": end_w * STRIDE,
            }
            for idx, (start_w, end_w, length) in enumerate(events, start=1)
        ]
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print()
    print("Saved evaluation results to:")
    print(RESULTS_PATH)


if __name__ == "__main__":
    main()