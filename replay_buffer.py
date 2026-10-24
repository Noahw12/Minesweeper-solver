import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples


class SumTree:
    """Binary tree of priorities for proportional sampling in O(log n)."""
    def __init__(self, size):
        self.size = size
        self.capacity = 1 << (size - 1).bit_length()
        self.depth = self.capacity.bit_length() - 1
        self.tree = np.zeros(2 * self.capacity, dtype=np.float64)

    @property
    def total(self):
        return self.tree[1]

    def leaf_values(self, leaves):
        return self.tree[leaves + self.capacity]

    def min_leaf(self, n_stored):
        return self.tree[self.capacity:self.capacity + n_stored].min()

    def update(self, leaves, values):
        if len(leaves) > 1:
            leaves, keep = np.unique(leaves[::-1], return_index=True)
            values = values[::-1][keep]
        nodes = leaves + self.capacity
        delta = values - self.tree[nodes]
        self.tree[nodes] = values
        #Distinct leaves share ancestors, so the additions must be unbuffered
        parents = nodes >> 1
        for _ in range(self.depth):
            np.add.at(self.tree, parents, delta)
            parents >>= 1

    def sample(self, batch_size):
        #Stratified, as in the PER paper
        segment = self.total / batch_size
        r = (np.arange(batch_size) + np.random.random(batch_size)) * segment
        nodes = np.ones(batch_size, dtype=np.int64)
        for _ in range(self.depth):
            left = nodes << 1
            left_sum = self.tree[left]
            go_right = r > left_sum
            r -= np.where(go_right, left_sum, 0.0)
            nodes = left + go_right
        assert self.tree[nodes].min() > 0, "sum tree reached a slot with no priority"
        return nodes - self.capacity


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized experience replay with optional multi-step returns."""
    def __init__(self, buffer_size, observation_space, action_space, device='auto', n_envs=1,
                 alpha=0.6, beta_initial=0.4, beta_final=1.0, n_steps=1, gamma=0.99, **kwargs):
        super().__init__(buffer_size, observation_space=observation_space, action_space=action_space,
                         device=device, n_envs=n_envs, **kwargs)
        if self.optimize_memory_usage:
            raise ValueError('PrioritizedReplayBuffer needs optimize_memory_usage=False')
        self.alpha = alpha
        self.beta_initial = beta_initial
        self.beta_final = beta_final
        self.beta = beta_initial
        #Empty slots stay at priority 0 and are never drawn
        self.tree = SumTree(self.buffer_size * self.n_envs)
        self.max_priority = 1.0
        self.n_steps = n_steps
        self.gamma = gamma

    def update_beta(self, progress_remaining):
        #SB3 lets progress_remaining drift outside [0, 1] at the end of a run
        progress_remaining = min(max(progress_remaining, 0.0), 1.0)
        self.beta = self.beta_final + (self.beta_initial - self.beta_final) * progress_remaining

    def add(self, obs, next_obs, action, reward, done, info):
        row = self.pos
        super().add(obs, next_obs, action, reward, done, info)
        leaves = row * self.n_envs + np.arange(self.n_envs)
        self.tree.update(leaves, np.full(self.n_envs, self.max_priority))

    def _sample_proportional(self, batch_size):
        leaves = self.tree.sample(batch_size)
        n_stored = (self.buffer_size if self.full else self.pos) * self.n_envs
        priorities = self.tree.leaf_values(leaves)
        weights = np.power(priorities / self.tree.min_leaf(n_stored), -self.beta)
        return leaves // self.n_envs, leaves % self.n_envs, weights.astype(np.float32)

    def n_step_returns(self, rows, envs):
        """Discounted sum of up to n_steps rewards from each sampled move"""
        returns = np.zeros(len(rows), dtype=np.float32)
        last = rows.copy()
        steps = np.zeros(len(rows), dtype=np.int64)
        going = np.ones(len(rows), dtype=bool)
        for k in range(self.n_steps):
            row = (rows + k) % self.buffer_size
            if k > 0:
                #Past the newest row lies unwritten or the oldest data
                going &= row != self.pos
            returns += np.where(going, self.gamma ** k * self.rewards[row, envs], 0.0)
            last = np.where(going, row, last)
            steps += going
            going &= self.dones[row, envs] == 0
        return returns, last, steps

    def sample(self, batch_size, env=None):
        batch_indices, env_indices, weights = self._sample_proportional(batch_size)
        returns, last, steps = self.n_step_returns(batch_indices, env_indices)
        data = (
            self._normalize_obs(self.observations[batch_indices, env_indices, :], env),
            self.actions[batch_indices, env_indices, :],
            self._normalize_obs(self.next_observations[last, env_indices, :], env),
            #A time limit is not a real end, so it still bootstraps
            (self.dones[last, env_indices] * (1 - self.timeouts[last, env_indices])).reshape(-1, 1),
            self._normalize_reward(returns.reshape(-1, 1), env),
        )
        return (ReplayBufferSamples(*tuple(map(self.to_torch, data))), batch_indices, env_indices,
                self.to_torch(weights), self.to_torch(steps.astype(np.float32)))

    def update_priorities(self, batch_indices, env_indices, td_errors):
        #The epsilon keeps a zero-error transition from never being drawn again
        new_priorities = np.power(np.abs(np.asarray(td_errors)) + 1e-6, self.alpha)
        leaves = np.asarray(batch_indices) * self.n_envs + np.asarray(env_indices)
        self.tree.update(leaves, new_priorities)
        self.max_priority = max(self.max_priority, float(new_priorities.max()))
