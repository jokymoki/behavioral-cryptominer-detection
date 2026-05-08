import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")

FIG_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SCORES_PATH = os.path.join(CKPT_DIR, "infected_scores.pt")
STATS_PATH = os.path.join(CKPT_DIR, "score_stats.json")

SCORE_KEY = "std_score_top5"
STRIDE = 10  # seconds between consecutive windows

def main():
    data = torch.load(SCORES_PATH, map_location="cpu", weights_only=False)

    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)

    if SCORE_KEY not in data:
        raise KeyError(f"Missing score key: {SCORE_KEY}")

    score = data[SCORE_KEY]
    if isinstance(score, torch.Tensor):
        score = score.numpy()

    threshold = stats["recommended_threshold"]

    x_windows = np.arange(len(score))
    x_seconds = x_windows * STRIDE

    print("Score key:", SCORE_KEY)
    print("Threshold:", threshold)
    print("Num windows:", len(score))
    print("Score min:", float(score.min()))
    print("Score max:", float(score.max()))

    # 1) Standard line plot
    plt.figure(figsize=(12, 4))
    plt.plot(x_seconds, score, label=SCORE_KEY)
    plt.axhline(threshold, linestyle="--", label=f"threshold = {threshold:.4f}")
    plt.title(f"Infected score series: {SCORE_KEY}")
    plt.xlabel("time (seconds)")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "infected_score_timeline.png"), dpi=300)
    plt.show()

    # 2) Log-scale plot for visibility
    plt.figure(figsize=(12, 4))
    plt.plot(x_seconds, score, label=SCORE_KEY)
    plt.axhline(threshold, linestyle="--", label=f"threshold = {threshold:.4f}")
    plt.yscale("log")
    plt.title(f"Infected score series (log scale): {SCORE_KEY}")
    plt.xlabel("time (seconds)")
    plt.ylabel("score (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "infected_score_timeline_log.png"), dpi=300)
    plt.show()

    # 3) Histogram of infected scores
    plt.figure(figsize=(8, 5))
    plt.hist(score, bins=50)
    plt.axvline(threshold, linestyle="--", label=f"threshold = {threshold:.4f}")
    plt.title(f"Infected score distribution: {SCORE_KEY}")
    plt.xlabel("score")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "infected_score_hist.png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    main()