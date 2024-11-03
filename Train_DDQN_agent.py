import gymnasium
from DDQN import DDQN, PrioritizedReplayBuffer, FeaturesExtractorBeginner
from stable_baselines3.dqn.policies import CnnPolicy
from Minesweeper_env import MinesweeperEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

def lr_schedule(progress_remaining):
    return 0.0001 + (0.0005 - 0.0001) * progress_remaining

def make_env():
    env = gymnasium.make('Minesweeper-v0')  
    return env 

buffer_size = 50000
batch_size = 64
gamma = 0.99
expl_fract = 0.9
expl_init = 1.0
expl_fin = 0.01
alpha = 0.6
timesteps = 500000 + buffer_size
n_envs = 16
lr = lr_schedule
tau = 0.005
train_freq = (1, "step")
gradient_steps = 8

vec_env = make_vec_env(make_env, n_envs=n_envs)

new_logger = configure("./logs/DDQN_Minesweeper/", ["stdout", "tensorboard"])

model = DDQN(
    CnnPolicy, 
    vec_env, 
    policy_kwargs={ 
    'features_extractor_class': FeaturesExtractorBeginner,
    'features_extractor_kwargs': {'features_dim': 64},
    'normalize_images': False,
    },
    learning_rate=lr, 
    verbose=1, 
    tensorboard_log="./logs/DDQN_Minesweeper/",  
    buffer_size=buffer_size, 
    batch_size=batch_size, 
    gamma=gamma, 
    exploration_fraction=expl_fract,
    exploration_initial_eps=expl_init, 
    exploration_final_eps=expl_fin, 
    replay_buffer_class=PrioritizedReplayBuffer,
    replay_buffer_kwargs={'alpha': alpha},
    learning_starts=buffer_size / 2,
    gradient_steps=gradient_steps,
    train_freq=train_freq,
    tau=tau
    )

model.learn(total_timesteps=timesteps, tb_log_name="DDQN_Minesweeper")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10000)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

model_location = "models/DDQN_agent_0.3"
model.save(model_location)
print(f"Saved model at {model_location}")
       
#TODO
#1. Check for bugs (Testa koden utan vec_env)
#2. Reward structure
#3. tune parameters
#4. implement logging
#5. clean up code (and add colored prints for training)
#6. Improve PER (SumTree)

