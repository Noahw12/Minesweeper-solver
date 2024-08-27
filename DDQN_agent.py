import gymnasium
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from Minesweeper_env import MinesweeperEnv
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import CnnPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
from stable_baselines3.common.utils import polyak_update

class QNetworkBeginner(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(QNetworkBeginner,self).__init__(observation_space, features_dim)
      
        self.conv = nn.Sequential(
            nn.Conv2d(10,features_dim, kernel_size=3, padding=1),
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
        return self.conv(observation)
        

#Intermediate and Expert networks not implemented yet
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
            
    def forward(self, observation, *args, **kwargs):
        
        obs_tensor = th.as_tensor(observation, device=self.device, dtype=th.int64)
        one_hot_obs = F.one_hot(obs_tensor, 10).permute(0, 3, 1, 2)
        #print(one_hot_obs.shape)
        
        q_values = self.q_net(one_hot_obs)
        action_mask = (obs_tensor == 9).flatten(start_dim=1)

        masked_q_values = th.where(action_mask, q_values, th.tensor(float('-inf')).to(q_values.device))
        return masked_q_values 
    
    def predict(self, observation, state=None, episode_start=None, deterministic=True): 
        q_values = self.forward(observation)
        action = th.argmax(q_values, dim=1)
        return action.cpu().numpy(), state

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
            
            with th.no_grad():
                #Compute the next Q-values using the target network
                one_hot_next_obs = F.one_hot(replay_data.next_observations.to(th.int64), 10).permute(0, 3, 1, 2)
                next_q_values = self.q_net_target(one_hot_next_obs)
                next_q_values_online = self.q_net(one_hot_next_obs)
                next_actions = next_q_values_online.argmax(dim=1, keepdim=True)
                next_q_values = th.gather(next_q_values, dim=1, index=next_actions).squeeze(1)
                #Expected Q values
                rewards = replay_data.rewards.squeeze(1)
                dones =  replay_data.dones.squeeze(1)
                target_q_values = rewards + (1 - dones ) * self.gamma * next_q_values
            
            #Get current Q-values estimates
            one_hot_obs = F.one_hot(replay_data.observations.to(th.int64), 10).permute(0, 3, 1, 2)
            current_q_values = self.q_net(one_hot_obs)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long()).squeeze(1)
        
            # Compute TD-error for updating the priorities in the PER buffer
            td_errors = th.abs(current_q_values - target_q_values).detach().cpu().numpy()
            
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
        super(PrioritizedReplayBuffer, self).__init__(buffer_size, observation_space=observation_space, 
                                                      action_space=action_space, device=device, n_envs=n_envs, **kwargs)
        self.alpha = alpha  #Controls the amount of prioritization used 
        self.priorities = np.zeros((max(buffer_size // n_envs, 1), n_envs), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, info):
        super().add(obs, next_obs, action, reward, done, info)
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

def make_env():
    env = gymnasium.make('Minesweeper-v0')  
    return env 

buffer_size = 10000
batch_size = 64
gamma = 0.99
expl_fract = 0.7
expl_init = 1.0
expl_fin = 0.05
alpha = 0.6
timesteps = 100000 + buffer_size
envs = 16
lr = lr_schedule
tau = 1
target_update_interval = 2048
train_freq = (1, "step")
gradient_steps = 8

vec_env = make_vec_env(make_env, n_envs=envs)

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
    learning_starts=buffer_size,
    target_update_interval=target_update_interval,
    gradient_steps=gradient_steps,
    train_freq=train_freq,
    tau=tau
    )

model.learn(total_timesteps=timesteps)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10000)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

#TODO
#4. Reward structure