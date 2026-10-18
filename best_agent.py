import os
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from agent import train

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VIDEO_DIR = os.path.join(RESULTS_DIR, "video")
os.makedirs(VIDEO_DIR, exist_ok=True)

EPISODES = 200_000
SEEDS = [0, 1, 2, 3, 4]

# EDIT THIS: baseline plus the settings that looked best in D, E and F.
CANDIDATES = {
    "baseline":         dict(alpha=0.1, gamma=1.0,  epsilon_decay=0.99995),
    "gamma0.99":        dict(alpha=0.1, gamma=0.99, epsilon_decay=0.99995),
    "fast_decay":       dict(alpha=0.1, gamma=1.0,  epsilon_decay=0.9999),
    "alpha0.5":         dict(alpha=0.5, gamma=1.0,  epsilon_decay=0.99995),
}
BASE = dict(epsilon=1.0, epsilon_min=0.05)

def evaluate(agent, n_episodes, seed):
    """Greedy evaluation: no exploration, no learning."""
    env = gym.make("Blackjack-v1", sab=True)
    obs, _ = env.reset(seed=seed)
    rewards, steps = [], []
    for ep in range(n_episodes):
        if ep > 0:
            obs, _ = env.reset()
        done, total, n = False, 0.0, 0
        while not done:
            obs, r, term, trunc, _ = env.step(agent.act(obs, greedy=True))
            total += r
            n += 1
            done = term or trunc
        rewards.append(total)
        steps.append(n)
    env.close()
    return np.array(rewards), np.array(steps)

def summarize(r, st):
    return dict(mean=r.mean(), se=r.std(ddof=1) / np.sqrt(len(r)),
                win=(r > 0).mean(), draw=(r == 0).mean(), loss=(r < 0).mean(),
                steps=st.mean())

# 1) Train every candidate on every seed
agents = {}
for name, cfg in CANDIDATES.items():
    agents[name] = [train(episodes=EPISODES, seed=s, **BASE, **cfg)[0] for s in SEEDS]
    print("trained", name)

# 2) Select the best using one set of greedy hands (50,000 per seed)
sel = {}
for name, lst in agents.items():
    sel[name] = np.mean([evaluate(a, 50_000, seed=1000 + s)[0].mean()
                         for s, a in enumerate(lst)])
    print(f"selection  {name:12s} mean reward = {sel[name]:.4f}")
best = max(sel, key=sel.get)
print("Best configuration:", best)

# 3) Final evaluation on NEW hands (different seeds from selection)
lines = []
for name in ["baseline", best] if best != "baseline" else ["baseline"]:
    big_r, big_s, small_r, small_s = [], [], [], []
    for s, a in enumerate(agents[name]):
        r, st = evaluate(a, 100_000, seed=2000 + s)
        big_r.append(r); big_s.append(st)
        r30, st30 = evaluate(a, 30, seed=3000 + s)
        small_r.append(r30); small_s.append(st30)
    big = summarize(np.concatenate(big_r), np.concatenate(big_s))
    per_seed = [r.mean() for r in big_r]
    small = summarize(small_r[0], small_s[0])   # seed 0, 30 episodes
    lines.append(
        f"{name}: 500k hands (5 seeds x 100k): mean={big['mean']:.4f} "
        f"+/- {big['se']:.4f} SE | win={big['win']:.3f} draw={big['draw']:.3f} "
        f"loss={big['loss']:.3f} | steps/hand={big['steps']:.2f} | "
        f"per-seed means={np.round(per_seed, 4)}\n"
        f"    30-episode run (seed 0): mean={small['mean']:.3f} +/- {small['se']:.3f} SE, "
        f"win={small['win']:.2f}"
    )
text = "\n".join(lines)
print(text)
with open(os.path.join(RESULTS_DIR, "evaluation.txt"), "w") as f:
    f.write(f"Best configuration: {best}\n" + text + "\n")

# 4) Record the MP4 of the best agent (seed 0), greedy play
env = gym.make("Blackjack-v1", sab=True, render_mode="rgb_array")
env = RecordVideo(env, video_folder=VIDEO_DIR, name_prefix="best_agent",
                  episode_trigger=lambda e: True)
agent = agents[best][0]
obs, _ = env.reset(seed=7)
for ep in range(5):
    if ep > 0:
        obs, _ = env.reset()
    done = False
    while not done:
        obs, r, term, trunc, _ = env.step(agent.act(obs, greedy=True))
        done = term or trunc
env.close()
print("Videos saved in", VIDEO_DIR)