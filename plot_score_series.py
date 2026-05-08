import torch
import matplotlib.pyplot as plt
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")

FIG_DIR = os.path.join(PROJECT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SCORES_PATH = os.path.join(CKPT_DIR, "val_scores.pt")

data = torch.load(SCORES_PATH, map_location="cpu", weights_only=False)

global_mse = data["global_mse"].numpy()
vector_l2 = data["vector_l2"].numpy()
std_score = data["std_score"].numpy()

#Global MSE dstr
plt.figure(figsize=(12,4))
plt.plot(global_mse)
plt.title("Global MSE validation windows")
plt.xlabel("window index")
plt.ylabel("score")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "val_global_mse_series.png"), dpi=300)
plt.show()

#Vector l2 dstr
plt.figure(figsize=(12, 4))
plt.plot(vector_l2)
plt.title("Vector L2 validation windows")
plt.xlabel("window index")
plt.ylabel("score")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "val_vector_l2_series.png"), dpi=300)
plt.show()

#Stand pos score dstr
plt.figure(figsize=(12, 4))
plt.plot(std_score)
plt.title("Standartized positive score validation windows")
plt.xlabel("window index")
plt.ylabel("score")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "val_std_score_series.png"), dpi=300)
plt.show()

#Threshold
threshold = 1.46

plt.figure(figsize=(12, 4))
plt.plot(std_score)
plt.axhline(threshold, color="red")
plt.title("Standartized score with threshold")
plt.xlabel("window index")
plt.ylabel("score")
plt.savefig(os.path.join(FIG_DIR, "val_std_score_threshold.png"), dpi=300)
plt.show()