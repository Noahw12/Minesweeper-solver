import gymnasium
import torch.nn as nn
import torch
from copy import deepcopy
from Minesweeper_env import MaskedMinesweeperEnv
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.dqn.policies import DQNPolicy, CnnPolicy

class QNetworkBeginner(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(QNetworkBeginner,self).__init__(observation_space, features_dim)
      
        self.conv = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            #nn.ReLU(),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(8 * 8 * 256, 1024), #8x8 tiles, 128 filter from conv layer
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim)
        )   
    
    def forward(self, observation):
        return self.conv(observation['obs'].unsqueeze(1))

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
        
        q_values = self.q_net(obs_tensor)
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

def lr_schedule(progress_remaining):
    return 0.001 + (0.01 - 0.001) * progress_remaining

def make_masked_env():
    env = gymnasium.make('Minesweeper-v0')  
    return MaskedMinesweeperEnv(env)

vec_env = make_vec_env(make_masked_env, n_envs=16)

model = DQN(ActionMaskPolicy, vec_env, learning_rate=lr_schedule, verbose=True, buffer_size=75000, 
            batch_size=64, gamma=0.99, exploration_fraction=0.5                                                                                       , 
            exploration_initial_eps=1.0, exploration_final_eps=0.05)

model.learn(total_timesteps=1000000)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

#model.save("dqn_minesweeper")

#Exploration rate
#num channels
#More conv layers