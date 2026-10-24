import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN

from minesweeper import COVERED

#Q-value given to revealed tiles so they are never chosen
XL_NEG = -1e9


class Rainbow(DQN):
    """Double DQN with the Rainbow extensions: prioritized replay and multi-step
    returns (replay_buffer.py), distributional C51, dueling and noisy nets
    (networks.py). Adds action masking, symmetry augmentation and a symmetry
    consistency loss."""
    def __init__(self, *args, augment_symmetry=True, symmetry_loss_coef=0.0, n_steps=1, **kwargs):
        #The buffer builds the multi-step returns
        buffer_kwargs = dict(kwargs.get('replay_buffer_kwargs') or {})
        buffer_kwargs.update(n_steps=n_steps, gamma=kwargs.get('gamma', 0.99))
        kwargs['replay_buffer_kwargs'] = buffer_kwargs
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps
        self.augment_symmetry = augment_symmetry
        self.symmetry_loss_coef = symmetry_loss_coef

    @property
    def noisy(self):
        return getattr(self.policy.q_net.features_extractor, 'noisy', False)

    @property
    def distributional(self):
        return getattr(self.policy.q_net.features_extractor, 'n_atoms', 1) > 1

    def action_mask(self, observation):
        """True for covered tiles, the legal moves."""
        return (observation == COVERED).flatten(start_dim=1)

    def apply_action_mask(self, q_values, mask):
        return th.where(mask, q_values, th.full_like(q_values, XL_NEG))

    def _symmetries(self):
        """Flat tile permutations for each symmetry of the board, and their inverses."""
        if getattr(self, '_sym_cache', None) is None:
            rows, cols = self.observation_space.shape
            flat = th.arange(rows * cols).reshape(rows, cols)
            views = [flat, th.flip(flat, (0,)), th.flip(flat, (1,)), th.flip(flat, (0, 1))]
            if rows == cols:
                turned = th.rot90(flat, 1, (0, 1))
                views += [turned, th.flip(turned, (0,)), th.flip(turned, (1,)), th.flip(turned, (0, 1))]
            gather = th.stack([v.reshape(-1) for v in views]).to(self.device)
            self._sym_cache = (gather, th.argsort(gather, dim=1))
        return self._sym_cache

    def _augment(self, obs, next_obs, actions):
        """Applies an independently drawn symmetry to each transition."""
        gather, inverse = self._symmetries()
        rows, cols = self.observation_space.shape
        picks = th.randint(len(gather), (obs.shape[0],), device=obs.device)
        take = gather[picks]
        obs = th.gather(obs.reshape(obs.shape[0], -1), 1, take).reshape(-1, rows, cols)
        next_obs = th.gather(next_obs.reshape(next_obs.shape[0], -1), 1, take).reshape(-1, rows, cols)
        actions = th.gather(inverse[picks], 1, actions.long())
        return obs, next_obs, actions

    def _with_turned_copy(self, obs, network):
        """Per-tile output for the boards and for a randomly turned copy of each,
        mapped back onto the original tiles, from a single network call."""
        gather, inverse = self._symmetries()
        rows, cols = self.observation_space.shape
        #Never the identity
        picks = th.randint(1, len(gather), (obs.shape[0],), device=obs.device)
        turned = th.gather(obs.reshape(obs.shape[0], -1), 1, gather[picks]).reshape(-1, rows, cols)
        out, turned_out = network(th.cat([obs, turned])).chunk(2)
        back = inverse[picks].view(*inverse[picks].shape, *([1] * (out.dim() - 2)))
        return out, th.gather(turned_out, 1, back.expand_as(turned_out))

    @staticmethod
    def _project(rewards, discount, next_probs, support):
        """C51 projection of rewards + discount * Z onto the fixed support."""
        n_atoms = support.shape[0]
        v_min, v_max = support[0].item(), support[-1].item()
        delta = (v_max - v_min) / (n_atoms - 1)
        shifted = (rewards[:, None] + discount[:, None] * support[None, :]).clamp(v_min, v_max)
        b = (shifted - v_min) / delta
        lower, upper = b.floor().long(), b.ceil().long()
        #Landing exactly on an atom would give both neighbours zero weight
        lower[(upper > 0) & (lower == upper)] -= 1
        upper[(lower < n_atoms - 1) & (lower == upper)] += 1
        target = th.zeros_like(next_probs)
        target.scatter_add_(1, lower, next_probs * (upper.float() - b))
        target.scatter_add_(1, upper, next_probs * (b - lower.float()))
        return target

    def _random_valid_actions(self, obs_tensor):
        mask = self.action_mask(obs_tensor)
        actions = np.zeros(mask.shape[0], dtype=np.int64)
        for i in range(mask.shape[0]):
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
        #A noisy target network draws its own noise for every batch
        self.policy.q_net_target.set_training_mode(self.noisy)

        losses, symmetry_losses = [], []
        for _ in range(gradient_steps):
            if self.noisy:
                self.policy.q_net.features_extractor.reset_noise()
                self.policy.q_net_target.features_extractor.reset_noise()
            replay_data, batch_indices, env_indices, weights, steps_used = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env)
            obs_tensor = replay_data.observations.to(th.int64)
            obs_tensor_next = replay_data.next_observations.to(th.int64)
            actions = replay_data.actions.long()

            if self.augment_symmetry:
                obs_tensor, obs_tensor_next, actions = self._augment(obs_tensor, obs_tensor_next, actions)

            loss_fn = self._distributional_loss if self.distributional else self._td_loss
            loss, priorities, q_values, turned_q_values = loss_fn(
                obs_tensor, obs_tensor_next, actions, replay_data, steps_used, weights)
            self.replay_buffer.update_priorities(batch_indices, env_indices, priorities)
            losses.append(loss.item())

            if self.symmetry_loss_coef > 0:
                #A turned board is the same puzzle, so covered tiles should keep their Q-values
                covered = self.action_mask(obs_tensor).float()
                gap = F.smooth_l1_loss(q_values, turned_q_values, reduction='none')
                per_board = (gap * covered).sum(dim=1) / covered.sum(dim=1).clamp(min=1)
                symmetry_loss = (weights * per_board).mean()
                symmetry_losses.append(symmetry_loss.item())
                loss = loss + self.symmetry_loss_coef * symmetry_loss

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        self.logger.record('train/loss', np.mean(losses))
        if symmetry_losses:
            self.logger.record('train/symmetry_loss', np.mean(symmetry_losses))
        self.logger.record('train/per_beta', self.replay_buffer.beta)

    def _td_loss(self, obs_tensor, obs_tensor_next, actions, replay_data, steps_used, weights):
        """Double DQN Huber loss, returning |TD error| as the priorities."""
        with th.no_grad():
            next_mask = self.action_mask(obs_tensor_next)
            next_q_target = self.apply_action_mask(self.policy.q_net_target(obs_tensor_next), next_mask)
            next_q_online = self.apply_action_mask(self.policy.q_net(obs_tensor_next), next_mask)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_values = th.gather(next_q_target, dim=1, index=next_actions).squeeze(1)
            rewards = replay_data.rewards.squeeze(1)
            dones = replay_data.dones.squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma ** steps_used * next_q_values

        #Not masked: masking the stored action would give it a huge TD error
        turned_q_values = None
        if self.symmetry_loss_coef > 0:
            q_values, turned_q_values = self._with_turned_copy(obs_tensor, self.policy.q_net)
        else:
            q_values = self.policy.q_net(obs_tensor)
        current_q_values = q_values.gather(1, actions).squeeze(1)

        td_errors = th.abs(current_q_values - target_q_values).detach().cpu().numpy()
        elementwise_loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
        return (weights * elementwise_loss).mean(), td_errors, q_values, turned_q_values

    def _distributional_loss(self, obs_tensor, obs_tensor_next, actions, replay_data, steps_used, weights):
        """C51 cross-entropy with Double DQN action selection, returning the
        cross-entropy as the priorities, as in Rainbow."""
        online = self.policy.q_net.features_extractor
        target = self.policy.q_net_target.features_extractor
        rows = th.arange(actions.shape[0], device=actions.device)
        with th.no_grad():
            next_mask = self.action_mask(obs_tensor_next)
            next_actions = self.apply_action_mask(self.policy.q_net(obs_tensor_next), next_mask).argmax(dim=1)
            next_probs = target.logits(obs_tensor_next)[rows, next_actions].softmax(dim=-1)
            dones = replay_data.dones.squeeze(1)
            discount = (1 - dones) * self.gamma ** steps_used
            target_probs = self._project(replay_data.rewards.squeeze(1), discount, next_probs, online.support)

        turned_q_values = None
        if self.symmetry_loss_coef > 0:
            logits, turned_logits = self._with_turned_copy(obs_tensor, online.logits)
            turned_q_values = online.q_from_logits(turned_logits)
        else:
            logits = online.logits(obs_tensor)
        log_probs = logits[rows, actions.squeeze(1)].log_softmax(dim=-1)
        cross_entropy = -(target_probs * log_probs).sum(dim=-1)
        priorities = cross_entropy.detach().cpu().numpy()
        loss = (weights * cross_entropy).mean()
        return loss, priorities, online.q_from_logits(logits), turned_q_values

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """Masked greedy action. When exploring, noisy nets use fresh noise and
        epsilon-greedy picks only covered tiles."""
        self.policy.set_training_mode(False)
        obs_tensor, vectorized_env = self.policy.obs_to_tensor(observation)

        if not deterministic and np.random.rand() < self.exploration_rate:
            actions = self._random_valid_actions(obs_tensor)
        else:
            explore = self.noisy and not deterministic
            if explore:
                self.policy.q_net.set_training_mode(True)
                self.policy.q_net.features_extractor.reset_noise()
            with th.no_grad():
                q_values = self.policy.q_net(obs_tensor)
            if explore:
                self.policy.q_net.set_training_mode(False)
            masked_q_values = self.apply_action_mask(q_values, self.action_mask(obs_tensor))
            actions = masked_q_values.argmax(dim=1).cpu().numpy().astype(np.int64)

        if not vectorized_env:
            actions = actions.squeeze(axis=0)
        return actions, state
