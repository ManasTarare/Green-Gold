"""
rl_agent.py — Deep Q-Network (DQN) Agent for datacenter optimization.

Architecture:
  - Policy network: MLP (obs_dim → 128 → 64 → n_actions)
  - Target network: Periodic hard copy from policy net
  - Experience replay buffer for stable training
  - ε-greedy exploration with linear decay
"""

import random
import numpy as np
from collections import deque

# ── Try PyTorch; fall back to pure-numpy implementation ──────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ══════════════════════════════════════════════════════════════════
# REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════
class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ══════════════════════════════════════════════════════════════════
# PYTORCH DQN
# ══════════════════════════════════════════════════════════════════
if HAS_TORCH:

    class QNetwork(nn.Module):
        """MLP Q-value estimator."""

        def __init__(self, obs_dim, n_actions):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions),
            )

        def forward(self, x):
            return self.net(x)


    class DQNAgent:
        """Deep Q-Network agent with target network and experience replay."""

        def __init__(
            self,
            obs_dim,
            n_actions,
            lr=1e-3,
            gamma=0.95,
            batch_size=64,
            buffer_capacity=50_000,
            target_update_freq=100,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=2000,
        ):
            self.obs_dim    = obs_dim
            self.n_actions  = n_actions
            self.gamma      = gamma
            self.batch_size = batch_size
            self.target_update_freq = target_update_freq

            # Epsilon schedule
            self.epsilon       = epsilon_start
            self.epsilon_start = epsilon_start
            self.epsilon_end   = epsilon_end
            self.epsilon_decay_steps = epsilon_decay_steps

            # Networks
            self.device     = torch.device("cpu")
            self.policy_net = QNetwork(obs_dim, n_actions).to(self.device)
            self.target_net = QNetwork(obs_dim, n_actions).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.buffer    = ReplayBuffer(buffer_capacity)

            # Tracking
            self.steps_done     = 0
            self.train_steps    = 0
            self.episodes       = 0
            self.total_reward   = 0.0
            self.losses         = []
            self.episode_rewards = []
            self.action_counts  = [0] * n_actions

        def select_action(self, state):
            """ε-greedy action selection. Returns action index."""
            self.steps_done += 1

            if random.random() < self.epsilon:
                action = random.randint(0, self.n_actions - 1)
            else:
                with torch.no_grad():
                    s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.policy_net(s)
                    action = q_values.argmax(dim=1).item()

            self.action_counts[action] += 1
            return action

        def get_q_values(self, state):
            """Return Q-values for all actions given current state."""
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.policy_net(s).squeeze(0).cpu().numpy().tolist()

        def store_transition(self, state, action, reward, next_state, done):
            """Store experience in replay buffer."""
            self.buffer.push(state, action, reward, next_state, done)
            self.total_reward += reward

        def train_step(self):
            """One gradient update from replay buffer. Returns loss or None."""
            if len(self.buffer) < self.batch_size:
                return None

            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

            states_t      = torch.FloatTensor(states).to(self.device)
            actions_t     = torch.LongTensor(actions).to(self.device)
            rewards_t     = torch.FloatTensor(rewards).to(self.device)
            next_states_t = torch.FloatTensor(next_states).to(self.device)
            dones_t       = torch.FloatTensor(dones).to(self.device)

            # Current Q-values
            q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

            # Target Q-values
            with torch.no_grad():
                next_q = self.target_net(next_states_t).max(dim=1)[0]
                target  = rewards_t + self.gamma * next_q * (1 - dones_t)

            loss = nn.MSELoss()(q_values, target)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

            self.train_steps += 1

            # Update target network periodically
            if self.train_steps % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            loss_val = loss.item()
            self.losses.append(loss_val)
            if len(self.losses) > 500:
                self.losses.pop(0)

            return loss_val

        def decay_epsilon(self):
            """Linear epsilon decay."""
            frac = min(1.0, self.steps_done / max(1, self.epsilon_decay_steps))
            self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

        def end_episode(self, episode_reward):
            """Track episode completion."""
            self.episodes += 1
            self.episode_rewards.append(episode_reward)
            if len(self.episode_rewards) > 500:
                self.episode_rewards.pop(0)

        def get_stats(self):
            """Return agent statistics for API/dashboard."""
            return {
                "episodes":         self.episodes,
                "steps_done":       self.steps_done,
                "train_steps":      self.train_steps,
                "epsilon":          round(self.epsilon, 4),
                "total_reward":     round(self.total_reward, 3),
                "avg_reward":       round(self.total_reward / max(1, self.steps_done), 4),
                "buffer_size":      len(self.buffer),
                "buffer_capacity":  self.buffer.buffer.maxlen,
                "recent_loss":      round(np.mean(self.losses[-50:]), 6) if self.losses else 0.0,
                "action_counts":    self.action_counts,
                "reward_history":   self.episode_rewards[-60:],
                "loss_history":     self.losses[-60:],
                "network_arch":     f"{self.obs_dim}→128→64→{self.n_actions}",
            }

        def save(self, path="model.pt"):
            torch.save({
                "policy_net":  self.policy_net.state_dict(),
                "target_net":  self.target_net.state_dict(),
                "optimizer":   self.optimizer.state_dict(),
                "steps_done":  self.steps_done,
                "train_steps": self.train_steps,
                "episodes":    self.episodes,
                "epsilon":     self.epsilon,
                "total_reward": self.total_reward,
                "action_counts": self.action_counts,
            }, path)

        def load(self, path="model.pt"):
            import os
            if not os.path.exists(path):
                return False
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.policy_net.load_state_dict(ckpt["policy_net"])
            self.target_net.load_state_dict(ckpt["target_net"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.steps_done    = ckpt.get("steps_done", 0)
            self.train_steps   = ckpt.get("train_steps", 0)
            self.episodes      = ckpt.get("episodes", 0)
            self.epsilon       = ckpt.get("epsilon", self.epsilon_end)
            self.total_reward  = ckpt.get("total_reward", 0.0)
            self.action_counts = ckpt.get("action_counts", [0] * self.n_actions)
            return True

else:
    # ══════════════════════════════════════════════════════════════
    # FALLBACK: Pure-numpy DQN (no PyTorch required)
    # ══════════════════════════════════════════════════════════════

    class DQNAgent:
        """Simplified DQN using numpy (fallback when PyTorch unavailable)."""

        def __init__(self, obs_dim, n_actions, lr=1e-3, gamma=0.95,
                     batch_size=64, buffer_capacity=50_000,
                     target_update_freq=100,
                     epsilon_start=1.0, epsilon_end=0.05,
                     epsilon_decay_steps=2000, **kwargs):
            self.obs_dim    = obs_dim
            self.n_actions  = n_actions
            self.gamma      = gamma
            self.batch_size = batch_size
            self.lr         = lr
            self.target_update_freq = target_update_freq

            self.epsilon       = epsilon_start
            self.epsilon_start = epsilon_start
            self.epsilon_end   = epsilon_end
            self.epsilon_decay_steps = epsilon_decay_steps

            # Simple 2-layer MLP weights (Xavier init)
            scale1 = np.sqrt(2.0 / obs_dim)
            scale2 = np.sqrt(2.0 / 128)
            scale3 = np.sqrt(2.0 / 64)
            self.W1 = np.random.randn(obs_dim, 128).astype(np.float32) * scale1
            self.b1 = np.zeros(128, dtype=np.float32)
            self.W2 = np.random.randn(128, 64).astype(np.float32) * scale2
            self.b2 = np.zeros(64, dtype=np.float32)
            self.W3 = np.random.randn(64, n_actions).astype(np.float32) * scale3
            self.b3 = np.zeros(n_actions, dtype=np.float32)

            # Target weights
            self.tW1 = self.W1.copy(); self.tb1 = self.b1.copy()
            self.tW2 = self.W2.copy(); self.tb2 = self.b2.copy()
            self.tW3 = self.W3.copy(); self.tb3 = self.b3.copy()

            self.buffer = ReplayBuffer(buffer_capacity)

            self.steps_done     = 0
            self.train_steps    = 0
            self.episodes       = 0
            self.total_reward   = 0.0
            self.losses         = []
            self.episode_rewards = []
            self.action_counts  = [0] * n_actions

        def _forward(self, x, W1, b1, W2, b2, W3, b3):
            h1 = np.maximum(0, x @ W1 + b1)
            h2 = np.maximum(0, h1 @ W2 + b2)
            return h2 @ W3 + b3

        def _forward_with_cache(self, x, W1, b1, W2, b2, W3, b3):
            z1 = x @ W1 + b1; h1 = np.maximum(0, z1)
            z2 = h1 @ W2 + b2; h2 = np.maximum(0, z2)
            out = h2 @ W3 + b3
            return out, (x, z1, h1, z2, h2)

        def select_action(self, state):
            self.steps_done += 1
            if random.random() < self.epsilon:
                action = random.randint(0, self.n_actions - 1)
            else:
                q = self._forward(state.reshape(1, -1), self.W1, self.b1,
                                  self.W2, self.b2, self.W3, self.b3)
                action = int(np.argmax(q[0]))
            self.action_counts[action] += 1
            return action

        def get_q_values(self, state):
            q = self._forward(state.reshape(1, -1), self.W1, self.b1,
                              self.W2, self.b2, self.W3, self.b3)
            return q[0].tolist()

        def store_transition(self, state, action, reward, next_state, done):
            self.buffer.push(state, action, reward, next_state, done)
            self.total_reward += reward

        def train_step(self):
            if len(self.buffer) < self.batch_size:
                return None
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

            # Forward pass
            q_all, cache = self._forward_with_cache(
                states, self.W1, self.b1, self.W2, self.b2, self.W3, self.b3)
            q_vals = q_all[np.arange(self.batch_size), actions]

            # Target
            next_q = self._forward(next_states, self.tW1, self.tb1,
                                   self.tW2, self.tb2, self.tW3, self.tb3)
            target = rewards + self.gamma * np.max(next_q, axis=1) * (1 - dones)

            # Loss and backprop
            td_error = q_vals - target
            loss = float(np.mean(td_error ** 2))

            x, z1, h1, z2, h2 = cache
            dout = np.zeros_like(q_all)
            dout[np.arange(self.batch_size), actions] = 2 * td_error / self.batch_size

            dW3 = h2.T @ dout
            db3 = np.sum(dout, axis=0)
            dh2 = dout @ self.W3.T
            dh2[z2 <= 0] = 0

            dW2 = h1.T @ dh2
            db2 = np.sum(dh2, axis=0)
            dh1 = dh2 @ self.W2.T
            dh1[z1 <= 0] = 0

            dW1 = x.T @ dh1
            db1 = np.sum(dh1, axis=0)

            # Gradient clipping
            for g in [dW1, db1, dW2, db2, dW3, db3]:
                np.clip(g, -1.0, 1.0, out=g)

            self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
            self.W3 -= self.lr * dW3; self.b3 -= self.lr * db3

            self.train_steps += 1
            if self.train_steps % self.target_update_freq == 0:
                self.tW1 = self.W1.copy(); self.tb1 = self.b1.copy()
                self.tW2 = self.W2.copy(); self.tb2 = self.b2.copy()
                self.tW3 = self.W3.copy(); self.tb3 = self.b3.copy()

            self.losses.append(loss)
            if len(self.losses) > 500:
                self.losses.pop(0)
            return loss

        def decay_epsilon(self):
            frac = min(1.0, self.steps_done / max(1, self.epsilon_decay_steps))
            self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

        def end_episode(self, episode_reward):
            self.episodes += 1
            self.episode_rewards.append(episode_reward)
            if len(self.episode_rewards) > 500:
                self.episode_rewards.pop(0)

        def get_stats(self):
            return {
                "episodes":         self.episodes,
                "steps_done":       self.steps_done,
                "train_steps":      self.train_steps,
                "epsilon":          round(self.epsilon, 4),
                "total_reward":     round(self.total_reward, 3),
                "avg_reward":       round(self.total_reward / max(1, self.steps_done), 4),
                "buffer_size":      len(self.buffer),
                "buffer_capacity":  self.buffer.buffer.maxlen,
                "recent_loss":      round(float(np.mean(self.losses[-50:])), 6) if self.losses else 0.0,
                "action_counts":    self.action_counts,
                "reward_history":   self.episode_rewards[-60:],
                "loss_history":     self.losses[-60:],
                "network_arch":     f"{self.obs_dim}→128→64→{self.n_actions}",
            }

        def save(self, path="model.npz"):
            np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                     W3=self.W3, b3=self.b3,
                     tW1=self.tW1, tb1=self.tb1, tW2=self.tW2, tb2=self.tb2,
                     tW3=self.tW3, tb3=self.tb3,
                     steps_done=self.steps_done, train_steps=self.train_steps,
                     episodes=self.episodes, epsilon=self.epsilon,
                     total_reward=self.total_reward,
                     action_counts=np.array(self.action_counts))

        def load(self, path="model.npz"):
            import os
            if not os.path.exists(path):
                return False
            d = np.load(path)
            self.W1 = d["W1"]; self.b1 = d["b1"]
            self.W2 = d["W2"]; self.b2 = d["b2"]
            self.W3 = d["W3"]; self.b3 = d["b3"]
            self.tW1 = d["tW1"]; self.tb1 = d["tb1"]
            self.tW2 = d["tW2"]; self.tb2 = d["tb2"]
            self.tW3 = d["tW3"]; self.tb3 = d["tb3"]
            self.steps_done   = int(d["steps_done"])
            self.train_steps  = int(d["train_steps"])
            self.episodes     = int(d["episodes"])
            self.epsilon      = float(d["epsilon"])
            self.total_reward = float(d["total_reward"])
            self.action_counts = d["action_counts"].tolist()
            return True


# ══════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════
def train_agent(env, agent, num_steps=2000, log_every=100):
    """
    Train the DQN agent in the environment.

    Args:
        env: DatacenterEnv instance
        agent: DQNAgent instance
        num_steps: Total training steps
        log_every: Print progress every N steps

    Returns:
        dict: Training history
    """
    obs = env.reset()
    history = {"rewards": [], "losses": [], "epsilons": []}

    for step in range(1, num_steps + 1):
        action = agent.select_action(obs)
        next_obs, reward, done, info = env.step(action)
        agent.store_transition(obs, action, reward, next_obs, done)

        loss = agent.train_step()
        agent.decay_epsilon()

        history["rewards"].append(reward)
        history["losses"].append(loss if loss is not None else 0.0)
        history["epsilons"].append(agent.epsilon)

        obs = next_obs

        if step % log_every == 0:
            avg_r = np.mean(history["rewards"][-log_every:])
            avg_l = np.mean([l for l in history["losses"][-log_every:] if l > 0]) if any(l > 0 for l in history["losses"][-log_every:]) else 0
            print(f"  Step {step:>5}/{num_steps} | "
                  f"ε={agent.epsilon:.3f} | "
                  f"avg_reward={avg_r:.3f} | "
                  f"avg_loss={avg_l:.5f} | "
                  f"buffer={len(agent.buffer)}")

        if done:
            agent.end_episode(env.episode_reward)
            obs = env.reset()

    agent.end_episode(env.episode_reward)
    return history
