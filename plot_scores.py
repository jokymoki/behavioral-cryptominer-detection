import torch
import matplotlib.pyplot as plt
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
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
plt.show()

#Vector l2 dstr
plt.figure(figsize=(8, 5))
plt.hist(vector_l2, bins=50)
plt.title("Vector L2 Distribution")
plt.xlabel("score")
plt.ylabel("count")
plt.tight_layout()
plt.show()

#Stand pos score dstr
plt.figure(figsize=(8, 5))
plt.hist(std_score, bins=50)
plt.title("Standartized positive score distribution")
plt.xlabel("score")
plt.ylabel("count")
plt.tight_layout()
plt.show()
