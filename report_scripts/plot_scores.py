import torch
import matplotlib.pyplot as plt
import os
from pathlib import Path

PROJECT_DIR = str(Path(__file__).resolve().parents[1])
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")

FIG_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


SCORES_PATH = os.path.join(CKPT_DIR, "val_scores.pt")

data = torch.load(SCORES_PATH, map_location="cpu", weights_only=False)

global_mse = data["global_mse"].numpy()
vector_l2 = data["vector_l2"].numpy()
std_score = data["std_score"].numpy()

#Global MSE dstr
plt.figure(figsize=(8,5))
plt.hist(global_mse, bins=50)
plt.title("Global MSE distribution")
plt.xlabel("scores")
plt.ylabel("count")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "val_global_mse_hist.png"), dpi=300)
plt.show()

#Vector l2 dstr
plt.figure(figsize=(8, 5))
plt.hist(vector_l2, bins=50)
plt.title("Vector L2 Distribution")
plt.xlabel("score")
plt.ylabel("count")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "val_vector_l2_hist.png"), dpi=300)
plt.show()

#Stand pos score dstr
plt.figure(figsize=(8, 5))
plt.hist(std_score, bins=50)
plt.title("Standartized positive score distribution")
plt.xlabel("score")
plt.ylabel("count")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "val_std_score_hist.png"), dpi=300)
plt.show()
