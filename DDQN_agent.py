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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from stable_baselines3.common.utils import polyak_update

class QNetworkBeginner(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(QNetworkBeginner,self).__init__(observation_space, features_dim)
      
        self.conv = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(8 * 8 * 128, 512), #8x8 tiles, 128 filter from conv layer
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim)
        )   
    
    def forward(self, observation):
        print("Shape before unsqueeze:", observation.shape)
        return self.conv(observation.unsqueeze(1))

class ActionMaskPolicy(CnnPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, features_dim=128):
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
            nn.Linear(features_dim, action_space.n), #64 tiles
        )
        self.q_net_target = deepcopy(self.q_net)
            
    
    def forward(self, observation, *args, **kwargs):

        obs_tensor = {'obs': torch.as_tensor(observation['obs'], device=self.device, dtype=torch.float32), 
                      'action_mask': torch.as_tensor(observation['action_mask'], device=self.device, dtype=torch.bool)}
        
        q_values = self.q_net(obs_tensor['obs'])
        masked_q_values = torch.where(obs_tensor['action_mask'], q_values, torch.tensor(float('-inf')).to(q_values.device))
        return masked_q_values 
    
    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        obs_tensor = {'obs': torch.as_tensor(observation['obs'], device=self.device, dtype=torch.float32),
                      'action_mask': torch.as_tensor(observation['action_mask'], device=self.device, dtype=torch.bool)}
        
        q_values = self.forward(obs_tensor)
        action = torch.argmax(q_values, dim=1)
        return action.cpu().numpy(), state
    
    def set_training_mode(self, mode):
        self.train(mode)
    
class QNetworkIntermediate(BaseFeaturesExtractor):
    pass

class QNetworkExpert(BaseFeaturesExtractor):
    pass  

# class WarmUpCallback(BaseCallback):
#     def __init__(self, warm_up_steps, verbose=0):
#         super(WarmUpCallback, self).__init__(verbose)
#         self.warm_up_steps = warm_up_steps

#     def _on_training_start(self):
#         #Perform warm-up actions without training
#         for _ in range(self.warm_up_steps):
#             obs = self.model.env.observation_space.sample()
#             action, _ = self.model.predict(obs)
#             new_obs, reward, done, _ = self.model.env.step(action)
#             # Prepare the new observation in the required format
#             self.model.replay_buffer.add(obs, new_obs, action, reward, done)
         
            
#     def _on_step(self):
#         return super()._on_step()

class DDQN(DQN):
    def __init__(self, *args, **kwargs):
        super(DDQN, self).__init__(*args, **kwargs)
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.train()
        
        #Update learning rate according to lr schedule
        self._update_learning_rate(self.policy.optimizer)
        
        for _ in range(gradient_steps):
            #Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            with torch.no_grad():
                #Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_values_online.argmax(dim=1, keepdim=True)
                next_q_values = torch.gather(next_q_values, dim=1, index=next_actions).squeeze(1)
                
                #Expected Q values
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
            
            #Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = torch.gather(current_q_values, dim=1, index=replay_data.actions.long()).squeeze(1)
            
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
        self.priorities = np.zeros((self.buffer_size,), dtype=np.float32)

    #TODO this should calculate td-error
    def add(self, obs, next_obs, action, reward, done, info):
        super().add(obs['obs'], next_obs['obs'], action, reward, done, info)
        max_priority = self.priorities.max() if self.full else self.priorities[:self.pos].max()
        max_priority = max_priority if max_priority > 0 else 1.0  #Avoid 0 priority
        self.priorities[self.pos - 1] = max_priority  

    def _sample_proportional(self, batch_size):
        if self.full:
            #The buffer is full, we can sample from the entire buffer
            total_priority = self.priorities.sum()
            sampled_indices = np.random.choice(self.buffer_size, batch_size, p=self.priorities / total_priority)
        else:
            #The buffer is not full, sample only up to self.pos to avoid uninitialized data
            total_priority = self.priorities[:self.pos].sum()
            sampled_indices = np.random.choice(self.pos, batch_size, p=self.priorities[:self.pos] / total_priority)
        return sampled_indices

    def sample(self, batch_size, env=None):
        batch_indices = self._sample_proportional(batch_size)
        env_indices = np.random.randint(0, high=self.n_envs, size=(batch_size,))
        
        data = (
            self._normalize_obs(self.observations[batch_indices, env_indices, :], env),
            self.actions[batch_indices, env_indices, :],
            self._normalize_obs(self.next_observations[batch_indices, env_indices, :], env),
            (self.dones[batch_indices, env_indices] * (1 - self.timeouts[batch_indices, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_indices, env_indices].reshape(-1, 1), env),
        )  
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))  
        

def lr_schedule(progress_remaining):
    return 0.0001 + (0.001 - 0.0001) * progress_remaining

def make_masked_env():
    env = gymnasium.make('Minesweeper-v0')  
    return MaskedMinesweeperEnv(env)

vec_env = make_vec_env(make_masked_env, n_envs=16)

model = DDQN(ActionMaskPolicy, vec_env, learning_rate=lr_schedule, verbose=True, buffer_size=75000, 
            batch_size=64, gamma=0.99, exploration_fraction=0.4 , exploration_initial_eps=1.0, 
            exploration_final_eps=0.05, replay_buffer_class=PrioritizedReplayBuffer, replay_buffer_kwargs={'alpha':0.6})

#warm_up_callback = WarmUpCallback(32500)

model.learn(total_timesteps=100000)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1000)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

#TODO
#Prioritized experience replay
#Warmup callback

#POTENTIAL improvements
#Num channels
#more conv layers
#Normalize