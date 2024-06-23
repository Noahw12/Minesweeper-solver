import gymnasium
import Minesweeper_env
import torch.nn as nn
import torch
import numpy as np
from Minesweeper_env import MaskedMinesweeperEnv
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import DQNPolicy, CnnPolicy

class QNetworkBeginner(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim = 64):
        super(QNetworkBeginner,self).__init__(observation_space, features_dim)
      
        self.features_extractor = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8 * 32, 256), #8x8 tiles, 32 filter from conv layer
            nn.ReLU(),
            nn.Linear(256, features_dim) #64 tiles
        )
    
    def forward(self, observation):
        return self.features_extractor(observation["obs"].unsqueeze(1))

class ActionMaskPolicy(DQNPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, features_dim=64):
        super(ActionMaskPolicy, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=QNetworkBeginner,
            features_extractor_kwargs={'features_dim': features_dim},
        )
        #self.q_net = nn.Linear(features_dim, action_space.n)
    
    def forward(self, obs, *args, **kwargs):
        print(self.features_extractor)
        if self.features_extractor is None:
            raise Exception("Feature extractor not initialized!")
        features = self.extract_features(obs['obs'])
        q_values = self.q_net(features)
        action_mask = obs['action_mask']
        masked_q_values = torch.where(action_mask, q_values, torch.tensor(float('-inf')).to(q_values.device))
        return masked_q_values 
    
    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        q_values = self.forward(observation)
        action = torch.argmax(q_values, dim=1)
        return action
    
    def set_training_mode(self, mode):
        # This correctly sets the training mode for all modules
        self.train(mode)
    
class QNetworkIntermediate(BaseFeaturesExtractor):
    pass

class QNetworkExpert(BaseFeaturesExtractor):
    pass  

def lr_schedule(progress_remaining):
    return 0.0001 + (0.001 - 0.0001) * progress_remaining
    
env = gymnasium.make('Minesweeper-v0')
masked_env = MaskedMinesweeperEnv(env)
masked_env = Monitor(masked_env)

model = DQN(ActionMaskPolicy, masked_env, learning_rate=lr_schedule, verbose=1, buffer_size=10000, 
            batch_size=32, gamma=0.99, exploration_fraction=0.1, 
            exploration_initial_eps=1.0, exploration_final_eps=0.1)

model.learn(total_timesteps=30000)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

model.save("dqn_minesweeper")