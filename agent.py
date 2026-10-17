import numpy as np
import gymnasium as gym


class QLearningAgent:
    def __init__(self, n_actions=2, alpha=0.1, gamma=1.0,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.99995, seed=0):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((32, 11, 2, n_actions))

    @staticmethod
    def _key(obs):
        player, dealer, ace = obs
        return (player, dealer, int(ace))

    def act(self, obs, greedy=False):
        s = self._key(obs)
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        q = self.Q[s]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def update(self, obs, a, r, next_obs, terminated):
        s = self._key(obs)
        if terminated:
            target = r
        else:
            target = r + self.gamma * self.Q[self._key(next_obs)].max()
        self.Q[s + (a,)] += self.alpha * (target - self.Q[s + (a,)])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(episodes=200_000, seed=0, **agent_kwargs):
    env = gym.make("Blackjack-v1", sab=True)
    agent = QLearningAgent(seed=seed, **agent_kwargs)
    rewards = []
    obs, _ = env.reset(seed=seed)
    for ep in range(episodes):
        if ep > 0:
            obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.update(obs, action, reward, next_obs, terminated)
            obs = next_obs
            total += reward
            done = terminated or truncated
        agent.decay_epsilon()
        rewards.append(total)
    env.close()
    return agent, np.array(rewards)


if __name__ == "__main__":
    agent, rewards = train()
    print("Mean reward, last 10000 episodes:", rewards[-10000:].mean())