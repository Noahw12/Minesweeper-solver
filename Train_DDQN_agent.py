import gymnasium
from DDQN import DDQN, PrioritizedReplayBuffer, MinesweeperCNN, MinesweeperPolicy
#Importing the module is what registers the Minesweeper env ids with gymnasium
from Minesweeper_env import MinesweeperEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

env_id = 'Minesweeper-v0'

def lr_schedule(progress_remaining):
    return 0.0001 + (0.0005 - 0.0001) * progress_remaining

def make_env():
    env = gymnasium.make(env_id)
    return env

buffer_size = 50000
batch_size = 64
#Minesweeper value is dominated by whether this click kills you. A far
#sighted discount spreads credit over a horizon the agent never survives.
gamma = 0.9
expl_fract = 0.2
expl_init = 1.0
expl_fin = 0.01
alpha = 0.6
beta_initial = 0.4
timesteps = 1000000 + buffer_size
n_envs = 16
lr = lr_schedule
tau = 0.005
#The target network is soft updated, so it has to be updated every step. With the
#SB3 default of 10000 a tau this small leaves the target network almost frozen.
target_update_interval = 1
train_freq = (1, "step")
gradient_steps = 8
learning_starts = buffer_size // 2

vec_env = make_vec_env(make_env, n_envs=n_envs)

new_logger = configure("./logs/DDQN_Minesweeper/", ["stdout", "tensorboard"])

model = DDQN(
    MinesweeperPolicy,
    vec_env,
    policy_kwargs={
    'features_extractor_class': MinesweeperCNN,
    'features_extractor_kwargs': {'n_filters': 64},
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
    replay_buffer_kwargs={'alpha': alpha, 'beta_initial': beta_initial},
    learning_starts=learning_starts,
    gradient_steps=gradient_steps,
    train_freq=train_freq,
    target_update_interval=target_update_interval,
    tau=tau
    )

model.set_logger(new_logger)

model.learn(total_timesteps=timesteps, tb_log_name="DDQN_Minesweeper")

#Evaluate on a single fresh environment, greedily
eval_env = make_vec_env(make_env, n_envs=1)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000, deterministic=True)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

model_location = "models/DDQN_agent_0.6"
model.save(model_location)
print(f"Saved model at {model_location}")

#TODO
#1. Reward structure
#2. tune parameters
#3. clean up code (and add colored prints for training)
#4. Improve PER (SumTree)
#5. Log the win rate, mean reward alone hides how many boards are actually cleared
