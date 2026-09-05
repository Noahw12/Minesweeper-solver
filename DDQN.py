import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from Minesweeper import COVERED
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples

#Very big negative number used for action masking
# '-inf' caused numerical instability
XL_NEG = -1e9

#Added to every TD-error before it becomes a priority, so that a perfectly
#predicted transition keeps a small chance of being sampled again
PER_EPSILON = 1e-6

class MinesweeperCNN(BaseFeaturesExtractor):
    #The last convolution produces one value per tile, so features_dim is fixed by
    #the board size and n_filters only sets the width of the hidden layers. Combined
    #with net_arch=[] this keeps one output of the network per action.
    def __init__(self, observation_space, n_filters=64):
        rows, cols = observation_space.shape
        super(MinesweeperCNN, self).__init__(observation_space, features_dim=rows * cols)

        #One input channel per tile value the agent can observe
        self.num_classes = int(observation_space.high.max()) + 1

        self.conv = nn.Sequential(
            nn.Conv2d(self.num_classes, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_filters, 1, kernel_size=1, padding=0),
            nn.Flatten()
        )

    def forward(self, observation):
        obs_tensor = observation.to(th.int64)
        one_hot_obs = F.one_hot(obs_tensor, self.num_classes).permute(0, 3, 1, 2).float()
        return self.conv(one_hot_obs)

#The feature extractor already outputs one Q-value per tile, so the network needs
#no head at all. Any dense head mixes every tile into every action and forces the
#agent to relearn the same local pattern for all 64 board positions, an identity
#head keeps the whole network translation equivariant.
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
        #Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return ConvQNetwork(**net_args).to(self.device)

class DDQN(DQN):
    def __init__(self, *args, **kwargs):
        super(DDQN, self).__init__(*args, **kwargs)

    #True for every tile that is still covered, those are the legal actions
    def action_mask(self, observation):
        return (observation == COVERED).flatten(start_dim=1)

    def apply_action_mask(self, q_values, observation):
        mask = self.action_mask(observation)
        return th.where(mask, q_values, th.full_like(q_values, XL_NEG))

    #Uniform sample over the legal actions of each observation in the batch
    def _random_valid_actions(self, obs_tensor):
        mask = self.action_mask(obs_tensor)
        actions = np.zeros(mask.shape[0], dtype=np.int64)
        for i in range(mask.shape[0]):
            valid_actions = mask[i].nonzero(as_tuple=False).squeeze(-1)
            if len(valid_actions) == 0:
                #Nothing left to clear, the episode is about to end anyway
                actions[i] = int(self.action_space.sample())
            else:
                actions[i] = valid_actions[th.randint(0, len(valid_actions), (1,))].item()
        return actions

    #SB3 draws unmasked random actions during the warm-up phase, which fills the
    #replay buffer with moves the agent is never allowed to make. Mask those too.
    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.num_timesteps < learning_starts:
            obs_tensor, _ = self.policy.obs_to_tensor(self._last_obs)
            action = self._random_valid_actions(obs_tensor)
            return action, action
        return super()._sample_action(learning_starts, action_noise, n_envs)

    #batch_size = number of samples from the replay buffer
    def train(self, gradient_steps, batch_size=100):
        self.policy.set_training_mode(True)

        #Update learning rate according to lr schedule
        self._update_learning_rate(self.policy.optimizer)
        #Anneal the importance sampling exponent towards 1 over training
        self.replay_buffer.update_beta(self._current_progress_remaining)

        losses = []
        for _ in range(gradient_steps):
            #Sample replay buffer
            replay_data, batch_indices, env_indices, weights = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env)
            #Get Q-values estimates
            obs_tensor = replay_data.observations.to(th.int64)
            obs_tensor_next = replay_data.next_observations.to(th.int64)

            with th.no_grad():
                #Compute the next Q-values using the target network
                next_q_values = self.policy.q_net_target(obs_tensor_next)
                next_q_values_online = self.policy.q_net(obs_tensor_next)
                masked_next_q_values = self.apply_action_mask(next_q_values, obs_tensor_next)
                masked_next_q_values_online = self.apply_action_mask(next_q_values_online, obs_tensor_next)

                next_actions = masked_next_q_values_online.argmax(dim=1, keepdim=True)
                next_q_values = th.gather(masked_next_q_values, dim=1, index=next_actions).squeeze(1)
                #A finished board has no legal action left, its masked Q-value is
                #XL_NEG. Zero it so the mask can never leak into a target.
                has_valid_action = self.action_mask(obs_tensor_next).any(dim=1)
                next_q_values = th.where(has_valid_action, next_q_values, th.zeros_like(next_q_values))
                #Expected Q values
                rewards = replay_data.rewards.squeeze(1)
                dones = replay_data.dones.squeeze(1)
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

            #The Q-value of the action that was actually taken. It must not be
            #masked: masking it would turn a stored action into XL_NEG and produce
            #an enormous TD-error that then dominates the prioritised sampling.
            current_q_values = self.policy.q_net(obs_tensor)
            current_q_values = current_q_values.gather(1, replay_data.actions.long()).squeeze(1)

            #Compute TD-error for updating the priorities in the PER buffer
            td_errors = th.abs(current_q_values - target_q_values).detach().cpu().numpy()
            self.replay_buffer.update_priorities(batch_indices, env_indices, td_errors)

            #Compute Huber loss (less sensitive to outliers), weighted by the
            #importance sampling weights that correct the prioritisation bias
            elementwise_loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
            loss = (weights * elementwise_loss).mean()
            losses.append(loss.item())

            #Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            #Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps

        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        self.logger.record('train/loss', np.mean(losses))
        self.logger.record('train/per_beta', self.replay_buffer.beta)

    #Override to make sure that no invalid tiles
    #are selected during exploration
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        self.policy.set_training_mode(False)
        #Adds the batch dimension when a single observation is passed in
        obs_tensor, vectorized_env = self.policy.obs_to_tensor(observation)

        #Exploration: sample random valid actions
        if not deterministic and np.random.rand() < self.exploration_rate:
            actions = self._random_valid_actions(obs_tensor)
        #Exploitation: perform the valid action with the biggest q-value
        else:
            with th.no_grad():
                q_values = self.policy.q_net(obs_tensor)
            masked_q_values = self.apply_action_mask(q_values, obs_tensor)
            actions = masked_q_values.argmax(dim=1).cpu().numpy().astype(np.int64)

        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, state

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device='auto', n_envs=1,
                 alpha=0.6, beta_initial=0.4, beta_final=1.0, **kwargs):
        super(PrioritizedReplayBuffer, self).__init__(buffer_size, observation_space=observation_space,
                                                      action_space=action_space, device=device,
                                                      n_envs=n_envs, **kwargs)
        if self.optimize_memory_usage:
            raise ValueError('PrioritizedReplayBuffer needs optimize_memory_usage=False')

        self.alpha = alpha  #Controls the amount of prioritization used
        #Beta controls how strongly the importance sampling weights correct the
        #bias of that prioritization, it is annealed to beta_final during training
        self.beta_initial = beta_initial
        self.beta_final = beta_final
        self.beta = beta_initial
        self.priorities = np.zeros((max(buffer_size // n_envs, 1), n_envs), dtype=np.float32)

    def update_beta(self, progress_remaining):
        #SB3 lets progress_remaining drift slightly outside [0, 1] at the end of a
        #run, and a beta above beta_final would over-correct the weights
        progress_remaining = min(max(progress_remaining, 0.0), 1.0)
        self.beta = self.beta_final + (self.beta_initial - self.beta_final) * progress_remaining

    def add(self, obs, next_obs, action, reward, done, info):
        super().add(obs, next_obs, action, reward, done, info)
        if self.full:
            max_priority = self.priorities.max()
        elif self.pos > 0:
            max_priority = self.priorities[:self.pos].max()
        else:
            max_priority = 0.0
        max_priority = max_priority if max_priority > 0 else 1.0  #Avoid 0 priority
        #super().add() already advanced pos, so pos - 1 is the row just written
        self.priorities[self.pos - 1] = max_priority

    def _sample_proportional(self, batch_size):
        if self.full:
            #The buffer is full, we can sample from the entire buffer
            flat_priorities = self.priorities.flatten()
        else:
            #The buffer is not full, sample only up to self.pos to avoid uninitialized data
            flat_priorities = self.priorities[:self.pos].flatten()

        total_priority = flat_priorities.sum()
        if total_priority > 0:
            probabilities = flat_priorities / total_priority
        else:
            #Every stored transition is predicted perfectly, fall back to uniform
            probabilities = np.full(flat_priorities.size, 1.0 / flat_priorities.size)

        sampled_indices = np.random.choice(flat_priorities.size, batch_size, p=probabilities)

        #Importance sampling weights, normalized so they only ever scale a loss down
        weights = np.power(flat_priorities.size * probabilities[sampled_indices], -self.beta)
        weights = weights / weights.max()

        batch_indices = sampled_indices // self.n_envs
        env_indices = sampled_indices % self.n_envs

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

        #Compute the new priorities. The epsilon keeps a transition with a zero
        #TD-error sampleable, without it the transition is dropped forever.
        new_priorities = np.power(np.abs(td_errors) + PER_EPSILON, self.alpha)

        #Update the priorities array
        self.priorities[batch_indices, env_indices] = new_priorities
