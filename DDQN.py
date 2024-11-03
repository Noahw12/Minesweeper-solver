import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from Minesweeper_env import MinesweeperEnv
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import CnnPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples

#Very big negative number used for action masking
# '-inf' caused numerical instability
XL_NEG = -1e9

#Number of cell types visible to the agent
NUM_CLASSES = 10

#Number of samples from the replay buffer
NUM_PER_SAMPLES = 100

class FeaturesExtractorBeginner(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(FeaturesExtractorBeginner,self).__init__(observation_space, features_dim)
      
        self.conv = nn.Sequential(
            nn.Conv2d(NUM_CLASSES, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, features_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features_dim, features_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features_dim, features_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(features_dim, 1, kernel_size=1, padding=0),
            #nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )   
    
    def forward(self, observation):
        obs_tensor = observation.to(th.int64)
        one_hot_obs = F.one_hot(obs_tensor, NUM_CLASSES).permute(0, 3, 1, 2).float()
        return self.conv(one_hot_obs)

#TODO Intermediate and Expert networks not implemented yet
class FeaturesExtractorIntermediate(BaseFeaturesExtractor):
    pass

class FeaturesExtractorExpert(BaseFeaturesExtractor):
    pass  

class DDQN(DQN):
    def __init__(self, *args, **kwargs):
        super(DDQN, self).__init__(*args, **kwargs)
        
    def apply_action_mask(self, q_values, observation):
        
        action_mask = (observation == 9).flatten(start_dim=1)
        
        masked_q_values = th.where(action_mask, q_values, th.full_like(q_values, XL_NEG, device=q_values.device))
        return masked_q_values
    
    #batch_size = number of samples from the replay buffer
    def train(self, gradient_steps, batch_size = NUM_PER_SAMPLES):
        self.policy.set_training_mode(True)
        
        #Update learning rate according to lr schedule
        self._update_learning_rate(self.policy.optimizer)
        
        losses = []
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
                masked_next_q_values = self.apply_action_mask(next_q_values, obs_tensor_next)
                masked_next_q_values_online = self.apply_action_mask(next_q_values_online, obs_tensor_next)
                
                next_actions = masked_next_q_values_online.argmax(dim=1, keepdim=True)
                next_q_values = th.gather(masked_next_q_values, dim=1, index=next_actions).squeeze(1)
                #Expected Q values
                rewards = replay_data.rewards.squeeze(1)
                dones =  replay_data.dones.squeeze(1)
                target_q_values = rewards + (1 - dones ) * self.gamma * next_q_values
           
            current_q_values = self.policy.q_net(obs_tensor).flatten(start_dim=1)
            masked_current_q_values = self.apply_action_mask(current_q_values, obs_tensor)
            current_q_values = masked_current_q_values.gather(1, replay_data.actions.long()).squeeze(1)
        
            #Compute TD-error for updating the priorities in the PER buffer
            td_errors = th.abs(current_q_values - target_q_values).detach().cpu().numpy() 
            self.replay_buffer.update_priorities(batch_indices, env_indices, td_errors)
            
            #Compute Huber loss (less sensitive to outliers)
            loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())
            
            #Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            #Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        
        self._n_updates += self.gradient_steps 
    
    #Override to make sure that no invalid tiles 
    #are selected during exploration   
    def predict(self, observation, state = None, episode_start = None, deterministic = False):
        
        obs_tensor = th.as_tensor(observation).to(self.device)
        action_mask = (obs_tensor == 9).flatten(start_dim=1)
        #Exploration: sample random valid actions
        if not deterministic and np.random.rand() < self.exploration_rate:
            if self.policy.is_vectorized_observation(observation):
                n_batches = action_mask.size(0)
                action = np.zeros(n_batches, np.int32)
                #Loop over all envs and select a random action
                for i in range(n_batches):
                    valid_actions = action_mask[i].nonzero(as_tuple=False).squeeze(-1)
                    random_action = valid_actions[th.randint(0, len(valid_actions), (1,))].item()
                    action[i] = random_action
            else:
                valid_actions = action_mask.nonzero(as_tuple=False).squeeze(-1)
                action = valid_actions[th.randint(0, len(valid_actions), (1,))].item()     
        #Exploitation: perform the action with biggest q-value
        else:
            #Compute Q-values
            q_values = self.policy.q_net(obs_tensor).flatten(start_dim=1)
            # Apply action mask
            masked_q_values = self.apply_action_mask(q_values, obs_tensor)
            # Select the action with the highest Q-value among valid actions
            action = masked_q_values.argmax(dim=1).cpu().numpy()
            # Ensure actions are integers
            action = action.astype(int)
        
        return action, state

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
        batch_indices = np.asarray(batch_indices)
        env_indices = np.asarray(env_indices)
        td_errors = np.asarray(td_errors)

        # Compute the new priorities
        new_priorities = np.power(td_errors, self.alpha)

        # Update the priorities array
        self.priorities[batch_indices, env_indices] = new_priorities