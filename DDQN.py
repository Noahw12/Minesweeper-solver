import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from Minesweeper_env import MinesweeperEnv
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import CnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from stable_baselines3.common.utils import polyak_update

#Very big negative number used for action masking
# '-inf' caused numerical instability
XL_NEG = -1e9

#Number of cell types visible to the agent
NUM_CLASSES = 10

class QNetworkBeginner(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(QNetworkBeginner,self).__init__(observation_space, features_dim)
      
        self.conv = nn.Sequential(
            nn.Conv2d(NUM_CLASSES,features_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features_dim, features_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features_dim, features_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features_dim, features_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features_dim, 1, kernel_size=1, padding=0),
            nn.Flatten()
        )   
    
    def forward(self, observation):
        obs_tensor = observation.to(th.int64)
        one_hot_obs = F.one_hot(obs_tensor, NUM_CLASSES).permute(0, 3, 1, 2).float()
        return self.conv(one_hot_obs)
        

#TODO Intermediate and Expert networks not implemented yet
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
    
    def apply_action_mask(self, q_values, observation):
        
        action_mask = (observation == 9).flatten(start_dim=1)
        
        masked_q_values = th.where(action_mask, q_values, th.tensor(XL_NEG, device=q_values.device))
        return masked_q_values
        
            
    def forward(self, observation, deterministic=False, *args, **kwargs):
        
        q_values = self.q_net(observation)
        masked_q_values = self.apply_action_mask(q_values, observation)
        return masked_q_values 

class DDQN(DQN):
    def __init__(self, *args, **kwargs):
        super(DDQN, self).__init__(*args, **kwargs)
        self._n_updates = 0
    
    #batch_size = number of samples from the replay buffer
    def train(self, gradient_steps, batch_size = 100):
        self.policy.train()
        
        #Update learning rate according to lr schedule
        self._update_learning_rate(self.policy.optimizer)
        
        for _ in range(gradient_steps):
            #Sample replay buffer
            replay_data, batch_indices, env_indices = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            #Get Q-values estimates
            obs_tensor = replay_data.observations.to(th.int64)
            obs_tensor_next = replay_data.next_observations.to(th.int64)
            
            with th.no_grad():
                #Compute the next Q-values using the target network
                next_q_values = self.policy.q_net_target(obs_tensor_next).flatten(start_dim=1)
                next_q_values_online = self.policy.q_net(obs_tensor_next).flatten(start_dim=1)
                masked_next_q_values = self.policy.apply_action_mask(next_q_values, obs_tensor_next)
                masked_next_q_values_online = self.policy.apply_action_mask(next_q_values_online, obs_tensor_next)
                
                next_actions = masked_next_q_values_online.argmax(dim=1, keepdim=True)
                next_q_values = th.gather(masked_next_q_values, dim=1, index=next_actions).squeeze(1)
                #Expected Q values
                rewards = replay_data.rewards.squeeze(1)
                dones =  replay_data.dones.squeeze(1)
                target_q_values = rewards + (1 - dones ) * self.gamma * next_q_values
           
            current_q_values = self.policy.q_net(obs_tensor).flatten(start_dim=1)
            masked_current_q_values = self.policy.apply_action_mask(current_q_values, obs_tensor)
            current_q_values = masked_current_q_values.gather(1, replay_data.actions.long()).squeeze(1)
        
            #Compute TD-error for updating the priorities in the PER buffer
            td_errors = th.abs(current_q_values - target_q_values).detach().cpu().numpy() 
            self.replay_buffer.update_priorities(batch_indices, env_indices, td_errors)
            
            #Compute Huber loss (less sensitive to outliers)
            loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values)
            
            #Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()
        
        #Update target network
        self._n_updates += 1   
        if self.tau < 1.0:
            #Soft update at every training step
            polyak_update(self.policy.q_net.parameters(), self.policy.q_net_target.parameters(), self.tau)
        else:
            #Hard update at specified intervals
            if self._n_updates % self.target_update_interval == 0:
                polyak_update(self.policy.q_net.parameters(), self.policy.q_net_target.parameters(), self.tau)

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device, alpha, n_envs, **kwargs):
        super(PrioritizedReplayBuffer, self).__init__(buffer_size, observation_space=observation_space, 
                                                      action_space=action_space, device=device, n_envs=n_envs, **kwargs)
        self.alpha = alpha  #Controls the amount of prioritization used 
        self.priorities = np.zeros((max(buffer_size // n_envs, 1), n_envs), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, info):
        super().add(obs, next_obs, action, reward, done, info)
        max_priority = self.priorities.max() if self.full else self.priorities[:self.pos].max()
        max_priority = max_priority if max_priority > 0 else 1.0  #Avoid 0 priority
        idx = self.pos if not self.full else (self.pos - 1) % self.buffer_size
        self.priorities[idx] = max_priority  
    
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