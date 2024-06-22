import gymnasium
import Minesweeper_env
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.dqn.policies import DQNPolicy
import torch.nn as nn
import torch


class QNetworkBeginner(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(QNetworkBeginner,self).__init__(observation_space, features_dim)
      
        self.model = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8 * 32, 256), #8x8 tiles, 32 filter from conv layer
            nn.ReLU(),
            nn.Linear(256, features_dim) #64 tiles
        )
        
    # def forward(self, observation):
    #     obs = torch.tensor(observation['obs'], dtype=torch.int32).unsqueeze(0)
    #     return self.model(obs)
    
    def forward(self, observation):
        obs = observation['obs']
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.unsqueeze(0)  # Add batch dimension if missing
        return self.model(obs)

class ActionMaskPolicy(DQNPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, features_dim=64):
        super(ActionMaskPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=QNetworkBeginner,
            #features_extractor_kwargs={'features_dim': features_dim}
        )
        
    def forward(self, obs, *args, **kwargs):
        q_values = self.q_net(obs)
        masked_q_values = torch.where(obs['action_mask'], q_values, torch.tensor(float('-inf')).to(q_values.device))
        return masked_q_values
    
    def predict(self,observation, action_mask):
        q_values = self.forward(observation, action_mask)
        action = torch.argmax(q_values, dim=1)
        return action
    
class QNetworkIntermediate(BaseFeaturesExtractor):
    pass

class QNetworkExpert(BaseFeaturesExtractor):
    pass
    
    
env = gymnasium.make('Minesweeper-v0')
masked_env = Monitor(env)

policy = ActionMaskPolicy(masked_env.observation_space, masked_env.action_space, 
                          lr_schedule=constant_fn(1e-3))

model = DQN(policy, masked_env, verbose=1, buffer_size=10000, learning_rate=0.001, 
            batch_size=32, gamma=0.99, exploration_fraction=0.1, 
            exploration_initial_eps=1.0, exploration_final_eps=0.1,
            policy_kwargs={'features_dim': 64})

# eval_callback = EvalCallback(env, best_model_save_path='./logs/',
#                              log_path='./logs/', eval_freq=500,
#                              deterministic=True, render=False)

model.learn(total_timesteps=30000)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

model.save("dqn_minesweeper")