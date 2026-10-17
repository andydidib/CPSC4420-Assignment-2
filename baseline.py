import numpy as np
import matplotlib.pyplot as plt
import os
from agent import train

CONFIG = dict(alpha=0.1, gamma=1.0, epsilon=1.0,
              epsilon_min=0.05, epsilon_decay=0.99995)
EPISODES = 200_000
SEEDS = [0, 1, 2, 3, 4]
WINDOW = 5000

def moving_avg(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")

curves = []
for s in SEEDS:
    _, rewards = train(episodes=EPISODES, seed=s, **CONFIG)
    curves.append(moving_avg(rewards, WINDOW))
    print(f"seed {s}: last-{WINDOW} mean = {rewards[-WINDOW:].mean():.4f}")

curves = np.array(curves)
mean, std = curves.mean(axis=0), curves.std(axis=0)
x = np.arange(len(mean)) + WINDOW

plt.figure(figsize=(8, 5))
plt.plot(x, mean, label="Mean over 5 seeds")
plt.fill_between(x, mean - std, mean + std, alpha=0.25, label="±1 std")
plt.xlabel("Training episode")
plt.ylabel(f"Reward ({WINDOW}-episode moving average)")
plt.title("Q-learning on Blackjack: baseline")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
out_path = os.path.join(RESULTS_DIR, "baseline.png")
plt.savefig(out_path, dpi=150)
print("Saved", out_path)