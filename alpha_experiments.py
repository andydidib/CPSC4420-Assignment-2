import os
import numpy as np
import matplotlib.pyplot as plt
from agent import train

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Baseline settings, only alpha changes
BASE = dict(gamma=1.0, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.99995)
EPISODES = 200_000
SEEDS = [0, 1, 2, 3, 4]
WINDOW = 5000
ALPHAS = [0.1, 0.5, 0.9]

def moving_avg(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")

plt.figure(figsize=(8, 5))
for alpha in ALPHAS:
    curves, finals = [], []
    for s in SEEDS:
        _, rewards = train(episodes=EPISODES, seed=s, alpha=alpha, **BASE)
        curves.append(moving_avg(rewards, WINDOW))
        finals.append(rewards[-20_000:].mean())
    curves = np.array(curves)
    mean, std = curves.mean(axis=0), curves.std(axis=0)
    x = np.arange(len(mean)) + WINDOW
    plt.plot(x, mean, label=f"alpha = {alpha}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    print(f"alpha={alpha}: last-20k mean = {np.mean(finals):.4f} "
          f"(std across seeds {np.std(finals):.4f})")

plt.xlabel("Training episode")
plt.ylabel(f"Reward ({WINDOW}-episode moving average)")
plt.title("Effect of learning rate (alpha) on Q-learning, Blackjack")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "alpha_comparison.png")
plt.savefig(out_path, dpi=150)
print("Saved", out_path)
