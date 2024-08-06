import gymnasium
import numpy as np
import torch.nn as nn
import torch
from copy import deepcopy
from Minesweeper_env import MaskedMinesweeperEnv
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.dqn.policies import CnnPolicy
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from stable_baselines3.common.utils import polyak_update
import sys

class QNetworkBeginner(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(QNetworkBeginner,self).__init__(observation_space, features_dim)
      
        self.conv = nn.Sequential(
            nn.Conv2d(1,features_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features_dim, features_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features_dim, features_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features_dim, features_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )   
    
    def forward(self, observation):
        return self.conv(observation.unsqueeze(1))
    
class QNetworkIntermediate(BaseFeaturesExtractor):
    pass

class QNetworkExpert(BaseFeaturesExtractor):
    pass  

class ActionMaskPolicy(CnnPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, features_dim=64):
        super(ActionMaskPolicy, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=QNetworkBeginner,
            features_extractor_kwargs={'features_dim': features_dim},
        )
        self.features_extractor = QNetworkBeginner(observation_space, features_dim)
        self.q_net = nn.Sequential(
            self.features_extractor,
            nn.Conv2d(features_dim, 1, kernel_size=1, padding=0),
            nn.Flatten()
            
        )
        self.q_net_target = deepcopy(self.q_net)
            
    def forward(self, observation, *args, **kwargs):

        obs_tensor = {'obs': torch.as_tensor(observation['obs'], device=self.device, dtype=torch.float32), 
                      'action_mask': torch.as_tensor(observation['action_mask'], device=self.device, dtype=torch.bool)}
        
        q_values = self.q_net(obs_tensor['obs'])
        masked_q_values = torch.where(obs_tensor['action_mask'], q_values, torch.tensor(float('-inf')).to(q_values.device))
        print(masked_q_values.shape)
        print(type(masked_q_values))
        return masked_q_values 
    
    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        obs_tensor = {'obs': torch.as_tensor(observation['obs'], device=self.device, dtype=torch.float32),
                      'action_mask': torch.as_tensor(observation['action_mask'], device=self.device, dtype=torch.bool)}
        
        q_values = self.forward(obs_tensor)
        action = torch.argmax(q_values, dim=1)
        return action.cpu().numpy(), state
    
    def set_training_mode(self, mode):
        self.train(mode)

class DDQN(DQN):
    def __init__(self, *args, **kwargs):
        super(DDQN, self).__init__(*args, **kwargs)
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.train()
        
        #Update learning rate according to lr schedule
        self._update_learning_rate(self.policy.optimizer)
        
        for _ in range(gradient_steps):
            #Sample replay buffer
            replay_data, batch_indices, env_indices = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            with torch.no_grad():
                #Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_values_online.argmax(dim=1, keepdim=True)
                next_q_values = torch.gather(next_q_values, dim=1, index=next_actions).squeeze(1)
                #Expected Q values
                rewards = replay_data.rewards.squeeze(1)
                dones =  replay_data.dones.squeeze(1)
                target_q_values = rewards + (1 - dones ) * self.gamma * next_q_values
            
            #Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long()).squeeze(1)
        
            # Compute TD-error for updating the priorities in the PER buffer
            td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()
            
            self.replay_buffer.update_priorities(batch_indices, env_indices, td_errors)
            
            #Compute Huber loss (less sensitive to outliers)
            loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values)
            
            #Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()
        
        # Update target network
        if self.gradient_steps % self.target_update_interval == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device, alpha, n_envs, **kwargs):
        super(PrioritizedReplayBuffer, self).__init__(buffer_size, observation_space=observation_space['obs'], 
                                                      action_space=action_space, device=device, n_envs=n_envs, **kwargs)
        self.alpha = alpha  #Controls the amount of prioritization used 
        self.priorities = np.zeros((max(buffer_size // n_envs, 1), n_envs), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, info):
        super().add(obs['obs'], next_obs['obs'], action, reward, done, info)
        max_priority = self.priorities.max() if self.full else self.priorities[:self.pos].max()
        max_priority = max_priority if max_priority > 0 else 1.0  #Avoid 0 priority
        self.priorities[self.pos - 1] = max_priority  
    
    def _sample_proportional(self, batch_size):
        if self.full:
            #The buffer is full, we can sample from the entire buffer   
            flat_priorities = self.priorities.flatten()
            total_priority = flat_priorities.sum()
            sampled_indices = np.random.choice(self.buffer_size * self.n_envs, batch_size, p=flat_priorities / total_priority)
        else:
            #The buffer is not full, sample only up to self.pos to avoid uninitialized data
            flat_priorities = self.priorities[:self.pos].flatten()
            total_priority = flat_priorities.sum()
            sampled_indices = np.random.choice(self.pos * self.n_envs, batch_size, p=flat_priorities / total_priority)
            
        batch_indices = sampled_indices // self.n_envs
        env_indices = sampled_indices % self.n_envs
            
        return batch_indices, env_indices

    def sample(self, batch_size, env=None):
        batch_indices, env_indices = self._sample_proportional(batch_size)
        
        data = (
            self._normalize_obs(self.observations[batch_indices, env_indices, :], env),
            self.actions[batch_indices, env_indices, :],
            self._normalize_obs(self.next_observations[batch_indices, env_indices, :], env),
            (self.dones[batch_indices, env_indices] * (1 - self.timeouts[batch_indices, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_indices, env_indices].reshape(-1, 1), env),
        )  
        return ReplayBufferSamples(*tuple(map(self.to_torch, data))), batch_indices, env_indices
    
    def update_priorities(self, batch_indices, env_indices, td_errors):
        for batch_idx, env_idx, td_error in zip(batch_indices, env_indices, td_errors):
            self.priorities[batch_idx, env_idx] = td_error ** self.alpha
        
def lr_schedule(progress_remaining):
    return 0.00025 + (0.001 - 0.00025) * progress_remaining

def make_masked_env():
    env = gymnasium.make('Minesweeper-v0')  
    return MaskedMinesweeperEnv(env)

buffer_size = 100000
batch_size = 64
gamma = 0.99
expl_fract = 0.5
expl_init = 1.0
expl_fin = 0.05
alpha = 0.6
timesteps = 1100000
envs = 16
lr = lr_schedule
tau = 1

#TODO tune and use these parameters
target_update_interval = 0
train_freq = 0
gradient_steps = 0

vec_env = make_vec_env(make_masked_env, n_envs=envs)

model = DDQN(
    ActionMaskPolicy, 
    vec_env, 
    learning_rate=lr, 
    verbose=True, 
    buffer_size=buffer_size, 
    batch_size=batch_size, 
    gamma=gamma, 
    exploration_fraction=expl_fract,
    exploration_initial_eps=expl_init, 
    exploration_final_eps=expl_fin, 
    replay_buffer_class=PrioritizedReplayBuffer, 
    replay_buffer_kwargs={'alpha': alpha},
    #learning_starts=buffer_size,
    tau=tau
    )

model.learn(total_timesteps=timesteps)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10000)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# :param target_update_interval: update the target network every ``target_update_interval``
#         environment steps.
# :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
#         like ``(5, "step")`` or ``(2, "episode")``.
# :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
#         Set to ``-1`` means to do as many gradient steps as steps done in the environment
#         during the rollout.

#TODO
#1. nätverken i min actionmask policy är fel, fixa den klassen 
#   (behöver det normalizeras som med bilder?)
#2. Updating networks
#3. öka antalet channels? one hot encoding
#4. Reward structure