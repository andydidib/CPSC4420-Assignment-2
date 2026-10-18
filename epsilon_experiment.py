import os
import numpy as np
import matplotlib.pyplot as plt
from agent import train

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Baseline settings, only epsilon_decay changes (alpha back at 0.1, gamma 1.0)
BASE = dict(alpha=0.1, gamma=1.0, epsilon=1.0, epsilon_min=0.05)
EPISODES = 200_000
SEEDS = [0, 1, 2, 3, 4]
WINDOW = 5000
DECAYS = [0.9999, 0.99995, 0.99999]   # fast, baseline, slow

def moving_avg(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")

plt.figure(figsize=(8, 5))
for decay in DECAYS:
    curves, finals = [], []
    for s in SEEDS:
        _, rewards = train(episodes=EPISODES, seed=s, epsilon_decay=decay, **BASE)
        curves.append(moving_avg(rewards, WINDOW))
        finals.append(rewards[-20_000:].mean())
    curves = np.array(curves)
    mean, std = curves.mean(axis=0), curves.std(axis=0)
    x = np.arange(len(mean)) + WINDOW
    plt.plot(x, mean, label=f"epsilon_decay = {decay}")
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    print(f"decay={decay}: last-20k mean = {np.mean(finals):.4f} "
          f"(std across seeds {np.std(finals):.4f})")

plt.xlabel("Training episode")
plt.ylabel(f"Reward ({WINDOW}-episode moving average)")
plt.title("Effect of epsilon decay on Q-learning, Blackjack")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, "epsilon_decay_comparison.png")
plt.savefig(out_path, dpi=150)
print("Saved", out_path)