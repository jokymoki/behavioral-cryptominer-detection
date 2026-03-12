import os
import json
import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")

SCORES_PATH = os.path.join(CKPT_DIR, "val_scores_from_script.pt")
STATS_PATH = os.path.join(CKPT_DIR, "score_stats.json")

STRIDE = 10
MIN_CONSECUTIVE = 3

def find_consecutive_runs(flags):
    runs = []
    start = None


    for i, flag in enumerate(flags):
        if flag and start is None:
            start=i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    
    if start is not None:
        runs.append((start, len(flags) - 1))
    
    return runs

def filtered_runs_by_length(runs, min_length):
    filtered = []

    for start, end in runs:
        length = end - start + 1
        if length >= min_length:
            filtered.append((start, end, length))

    return filtered

def windows_index_to_seconds(idx, stride):
    return idx * stride

def main():
    data = torch.load(SCORES_PATH, map_location="cpu", weights_only=False)

    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)
    
    std_score = data["std_score"]
    threshold = stats["recommended_threshold"]

    if isinstance(std_score, torch.Tensor):
        std_score = std_score.numpy()
    
    print("Threshold:", threshold)
    print("Total windows:", len(std_score))

    above = std_score > threshold
    num_above = int(above.sum())

    print("Windows above threshold:", num_above)

    runs = find_consecutive_runs(above)
    events = filtered_runs_by_length(runs, MIN_CONSECUTIVE)

    if not events:
        print(f"No events with at least {MIN_CONSECUTIVE} consecutive windows.")
        return 

    print(f"\nDetected events (min {MIN_CONSECUTIVE} consecutive windows):")
    for idx, (start_w, end_w, length) in enumerate(events, start=1):
        start_sec = windows_index_to_seconds(start_w, STRIDE)
        end_sec = windows_index_to_seconds(end_w, STRIDE)

        print(
            f"event {idx}: windows {start_w}..{end_w}, "
            f"length={length}, "
            f"time={start_sec}s..{end_sec}s"
        )

if __name__ == "__main__":
    main()