import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = str(Path(__file__).resolve().parents[1])
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")

FIG_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SCORES_PATH = os.path.join(CKPT_DIR, "mixed_scores.pt")
STATS_PATH = os.path.join(CKPT_DIR, "score_stats.json")

SCORE_KEY = "std_score_top5"
STRIDE = 10

TRUE_INFECTED_START = 2000
TRUE_INFECTED_END = 4000

DETECTED_START = 1880
DETECTED_END = 3920


def main():
    data = torch.load(SCORES_PATH, map_location="cpu", weights_only=False)

    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)

    score = data[SCORE_KEY]
    if isinstance(score, torch.Tensor):
        score = score.cpu().numpy()

    threshold = stats["recommended_threshold"]

    x = np.arange(len(score)) * STRIDE

    print("Score key:", SCORE_KEY)
    print("Threshold:", threshold)
    print("Num windows:", len(score))
    print("Score min:", float(score.min()))
    print("Score max:", float(score.max()))

    plt.figure(figsize=(13, 5))
    plt.plot(x, score, label=SCORE_KEY)
    plt.axhline(threshold, linestyle="--", label=f"threshold = {threshold:.4f}")

    plt.axvspan(
        TRUE_INFECTED_START,
        TRUE_INFECTED_END,
        alpha=0.2,
        label="true infected segment"
    )

    plt.axvspan(
        DETECTED_START,
        DETECTED_END,
        alpha=0.2,
        label="detected event"
    )

    plt.title("Mixed scenario: anomaly score over time")
    plt.xlabel("time (seconds)")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "mixed_score_timeline.png"), dpi=300)
    plt.show()

    plt.figure(figsize=(13, 5))
    plt.plot(x, score, label=SCORE_KEY)
    plt.axhline(threshold, linestyle="--", label=f"threshold = {threshold:.4f}")

    plt.axvspan(
        TRUE_INFECTED_START,
        TRUE_INFECTED_END,
        alpha=0.2,
        label="true infected segment"
    )

    plt.axvspan(
        DETECTED_START,
        DETECTED_END,
        alpha=0.2,
        label="detected event"
    )

    plt.yscale("log")
    plt.title("Mixed scenario: anomaly score over time (log scale)")
    plt.xlabel("time (seconds)")
    plt.ylabel("score (log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "mixed_score_timeline_log.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
