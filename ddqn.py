import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from minesweeper import COVERED
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples

#Very big negative number used for action masking
XL_NEG = -1e9

#The skip is what keeps a stack this deep trainable.
class ResidualBlock(nn.Module):
    def __init__(self, n_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        return F.relu(x + self.conv2(h))

class MinesweeperResCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, n_filters=128, n_blocks=3):
        rows, cols = observation_space.shape
        super(MinesweeperResCNN, self).__init__(observation_space, features_dim=rows * cols)

        #One input channel per tile value the agent can observe
        self.num_classes = int(observation_space.high.max()) + 1

        layers = [nn.Conv2d(self.num_classes, n_filters, kernel_size=3, padding=1), nn.ReLU()]
        layers += [ResidualBlock(n_filters) for _ in range(n_blocks)]
        #Down to one value per tile, which is the Q-value of clearing it
        layers += [nn.Conv2d(n_filters, 1, kernel_size=1, padding=0), nn.Flatten()]
        self.conv = nn.Sequential(*layers)

    def forward(self, observation):
        obs_tensor = observation.to(th.int64)
        one_hot_obs = F.one_hot(obs_tensor, self.num_classes).permute(0, 3, 1, 2).float()
        return self.conv(one_hot_obs)

#Avoid standard implementations MLP which destroys spatial information of CNN
class ConvQNetwork(QNetwork):
    def __init__(self, *args, **kwargs):
        super(ConvQNetwork, self).__init__(*args, **kwargs)
        action_dim = int(self.action_space.n)
        if self.features_dim != action_dim:
            raise ValueError(f"ConvQNetwork needs one feature per action, got "
                             f"{self.features_dim} features for {action_dim} actions")
        self.q_net = nn.Identity()

class MinesweeperPolicy(DQNPolicy):
    def make_q_net(self):
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return ConvQNetwork(**net_args).to(self.device)

class DDQN(DQN):
    #True for every tile that is still covered, those are the legal actions
    def action_mask(self, observation):
        return (observation == COVERED).flatten(start_dim=1)

    #Takes the mask rather than the observation, so a caller that needs it more
    #than once builds it once
    def apply_action_mask(self, q_values, mask):
        return th.where(mask, q_values, th.full_like(q_values, XL_NEG))

    def _random_valid_actions(self, obs_tensor):
        mask = self.action_mask(obs_tensor)
        actions = np.zeros(mask.shape[0], dtype=np.int64)
        for i in range(mask.shape[0]):
            #Never empty: a mine is never revealed, so mines are always still covered
            valid_actions = mask[i].nonzero(as_tuple=False).squeeze(-1)
            actions[i] = valid_actions[th.randint(0, len(valid_actions), (1,))].item()
        return actions

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.num_timesteps < learning_starts:
            obs_tensor, _ = self.policy.obs_to_tensor(self._last_obs)
            action = self._random_valid_actions(obs_tensor)
            return action, action
        return super()._sample_action(learning_starts, action_noise, n_envs)

    def train(self, gradient_steps, batch_size=100):
        self.policy.set_training_mode(True)

        self._update_learning_rate(self.policy.optimizer)
        self.replay_buffer.update_beta(self._current_progress_remaining)

        losses = []
        for _ in range(gradient_steps):
            replay_data, batch_indices, env_indices, weights = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env)
            obs_tensor = replay_data.observations.to(th.int64)
            obs_tensor_next = replay_data.next_observations.to(th.int64)

            with th.no_grad():
                next_mask = self.action_mask(obs_tensor_next)
                next_q_values = self.policy.q_net_target(obs_tensor_next)
                next_q_values_online = self.policy.q_net(obs_tensor_next)
                masked_next_q_values = self.apply_action_mask(next_q_values, next_mask)
                masked_next_q_values_online = self.apply_action_mask(next_q_values_online, next_mask)

                next_actions = masked_next_q_values_online.argmax(dim=1, keepdim=True)
                next_q_values = th.gather(masked_next_q_values, dim=1, index=next_actions).squeeze(1)
                rewards = replay_data.rewards.squeeze(1)
                dones = replay_data.dones.squeeze(1)
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            #Must not be masked: that would turn a stored action into XL_NEG and
            #produce a huge TD-error that then dominates prioritised sampling.
            current_q_values = self.policy.q_net(obs_tensor)
            current_q_values = current_q_values.gather(1, replay_data.actions.long()).squeeze(1)

            td_errors = th.abs(current_q_values - target_q_values).detach().cpu().numpy()
            self.replay_buffer.update_priorities(batch_indices, env_indices, td_errors)

            #Huber loss, weighted to correct the prioritisation bias
            elementwise_loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
            loss = (weights * elementwise_loss).mean()
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps

        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        self.logger.record('train/loss', np.mean(losses))
        self.logger.record('train/per_beta', self.replay_buffer.beta)

    #Overridden so exploration never picks an already revealed tile
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        self.policy.set_training_mode(False)
        #Adds the batch dimension when a single observation is passed in
        obs_tensor, vectorized_env = self.policy.obs_to_tensor(observation)

        if not deterministic and np.random.rand() < self.exploration_rate:
            actions = self._random_valid_actions(obs_tensor)
        else:
            with th.no_grad():
                q_values = self.policy.q_net(obs_tensor)
            masked_q_values = self.apply_action_mask(q_values, self.action_mask(obs_tensor))
            actions = masked_q_values.argmax(dim=1).cpu().numpy().astype(np.int64)

        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, state

#SumTree implementation for PER, for effieciency
class SumTree:
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

        #Assert correctness
        assert self.tree[nodes].min() > 0, \
            "sum tree descent reached a slot with no priority, the node sums are wrong"
        return nodes - self.capacity

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device='auto', n_envs=1,
                 alpha=0.6, beta_initial=0.4, beta_final=1.0, **kwargs):
        super(PrioritizedReplayBuffer, self).__init__(buffer_size, observation_space=observation_space,
                                                      action_space=action_space, device=device,
                                                      n_envs=n_envs, **kwargs)
        if self.optimize_memory_usage:
            raise ValueError('PrioritizedReplayBuffer needs optimize_memory_usage=False')

        self.alpha = alpha
        #Beta sets how strongly the weights correct that bias
        self.beta_initial = beta_initial
        self.beta_final = beta_final
        self.beta = beta_initial
        #Empty slots stay at priority 0 and are never drawn
        self.tree = SumTree(self.buffer_size * self.n_envs)
        self.max_priority = 1.0

    def update_beta(self, progress_remaining):
        #SB3 lets progress_remaining drift outside [0, 1] at the end of a run
        progress_remaining = min(max(progress_remaining, 0.0), 1.0)
        self.beta = self.beta_final + (self.beta_initial - self.beta_final) * progress_remaining

    def add(self, obs, next_obs, action, reward, done, info):
        #Read the row before super().add() advances it
        row = self.pos
        super().add(obs, next_obs, action, reward, done, info)
        leaves = row * self.n_envs + np.arange(self.n_envs)
        self.tree.update(leaves, np.full(self.n_envs, self.max_priority))

    def _sample_proportional(self, batch_size):
        leaves = self.tree.sample(batch_size)
        n_stored = (self.buffer_size if self.full else self.pos) * self.n_envs
        
        priorities = self.tree.leaf_values(leaves)
        weights = np.power(priorities / self.tree.min_leaf(n_stored), -self.beta)

        batch_indices = leaves // self.n_envs
        env_indices = leaves % self.n_envs

        return batch_indices, env_indices, weights.astype(np.float32)

    def sample(self, batch_size, env=None):
        batch_indices, env_indices, weights = self._sample_proportional(batch_size)

        data = (
            self._normalize_obs(self.observations[batch_indices, env_indices, :], env),
            self.actions[batch_indices, env_indices, :],
            self._normalize_obs(self.next_observations[batch_indices, env_indices, :], env),
            (self.dones[batch_indices, env_indices] * (1 - self.timeouts[batch_indices, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_indices, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data))), batch_indices, env_indices, self.to_torch(weights)

    def update_priorities(self, batch_indices, env_indices, td_errors):
        batch_indices = np.asarray(batch_indices)
        env_indices = np.asarray(env_indices)
        td_errors = np.asarray(td_errors)

        #The epsilon stops a zero TD-error transition being dropped forever
        new_priorities = np.power(np.abs(td_errors) + 1e-6, self.alpha)

        leaves = batch_indices * self.n_envs + env_indices
        self.tree.update(leaves, new_priorities)
        self.max_priority = max(self.max_priority, float(new_priorities.max()))