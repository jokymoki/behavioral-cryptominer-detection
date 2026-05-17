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

VAL_SCORES_PATH = os.path.join(CKPT_DIR, "val_scores_from_script.pt")
INFECTED_SCORES_PATH = os.path.join(CKPT_DIR, "infected_scores.pt")
STATS_PATH = os.path.join(CKPT_DIR, "score_stats.json")

VAL_SCORE_KEY = "std_score"
INFECTED_SCORE_KEY = "std_score_top5"


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)


def main():
    val_data = torch.load(VAL_SCORES_PATH, map_location="cpu", weights_only=False)
    infected_data = torch.load(INFECTED_SCORES_PATH, map_location="cpu", weights_only=False)

    with open(STATS_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)

    if VAL_SCORE_KEY not in val_data:
        raise KeyError(f"Missing key in val file: {VAL_SCORE_KEY}")

    if INFECTED_SCORE_KEY not in infected_data:
        raise KeyError(f"Missing key in infected file: {INFECTED_SCORE_KEY}")

    val_score = to_numpy(val_data[VAL_SCORE_KEY])
    infected_score = to_numpy(infected_data[INFECTED_SCORE_KEY])

    threshold = stats["recommended_threshold"]

    print("Validation score key:", VAL_SCORE_KEY)
    print("Infected score key:", INFECTED_SCORE_KEY)
    print("Threshold:", threshold)

    print("\nValidation stats")
    print("count:", len(val_score))
    print("min:", float(val_score.min()))
    print("max:", float(val_score.max()))
    print("mean:", float(val_score.mean()))

    print("\nInfected stats")
    print("count:", len(infected_score))
    print("min:", float(infected_score.min()))
    print("max:", float(infected_score.max()))
    print("mean:", float(infected_score.mean()))

    # 1) Full-range histogram
    plt.figure(figsize=(10, 5))
    plt.hist(val_score, bins=50, alpha=0.7, label="normal validation")
    plt.hist(infected_score, bins=50, alpha=0.7, label="infected")
    plt.axvline(threshold, linestyle="--", label=f"threshold = {threshold:.4f}")
    plt.title("Normal vs infected score distribution")
    plt.xlabel("score")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "normal_vs_infected_hist.png"), dpi=300)
    plt.show()

    # 2) Log-x histogram for visibility
    val_score_safe = np.clip(val_score, 1e-6, None)
    infected_score_safe = np.clip(infected_score, 1e-6, None)

    bins = np.logspace(
        np.log10(min(val_score_safe.min(), infected_score_safe.min())),
        np.log10(max(val_score_safe.max(), infected_score_safe.max())),
        60
    )

    plt.figure(figsize=(10, 5))
    plt.hist(val_score_safe, bins=bins, alpha=0.7, label="normal validation")
    plt.hist(infected_score_safe, bins=bins, alpha=0.7, label="infected")
    plt.axvline(max(threshold, 1e-6), linestyle="--", label=f"threshold = {threshold:.4f}")
    plt.xscale("log")
    plt.title("Normal vs infected score distribution (log-x)")
    plt.xlabel("score (log scale)")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "normal_vs_infected_hist_log.png"), dpi=300)
    plt.show()

    # 3) Boxplot comparison
    plt.figure(figsize=(8, 5))
    plt.boxplot([val_score, infected_score], labels=["normal validation", "infected"], showfliers=True)
    plt.axhline(threshold, linestyle="--", label=f"threshold = {threshold:.4f}")
    plt.title("Normal vs infected score comparison")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "normal_vs_infected_boxplot.png"), dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
