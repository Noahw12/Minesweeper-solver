import gymnasium
from ddqn import DDQN, PrioritizedReplayBuffer, MinesweeperResCNN, MinesweeperPolicy
#Unused, but importing it registers the env ids with gymnasium
from minesweeper_env import MinesweeperEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

env_id = 'Minesweeper-v0'

lr_initial = 0.00025
lr_final = 0.00005

def lr_schedule(progress_remaining):
    return lr_final + (lr_initial - lr_final) * progress_remaining

def make_env():
    env = gymnasium.make(env_id)
    return env

buffer_size = 1000000
batch_size = 32
gamma = 0.9
expl_fract = 0.1
expl_init = 1.0
expl_fin = 0.01
learning_starts = (buffer_size // 10)

timesteps = 6000000
n_envs = 16
train_freq = (1, "step")
gradient_steps = 4

tau = 1.0 
target_update_interval = 10000

vec_env = make_vec_env(make_env, n_envs=n_envs)

new_logger = configure("./logs/DDQN_Minesweeper/", ["stdout", "tensorboard"])

model = DDQN(
    MinesweeperPolicy,
    vec_env,
    policy_kwargs={
    'features_extractor_class': MinesweeperResCNN,
    'features_extractor_kwargs': {'n_filters': 64, 'n_blocks': 3},
    'normalize_images': False,
    },
    learning_rate=lr_schedule,
    verbose=1,
    tensorboard_log="./logs/DDQN_Minesweeper/",
    buffer_size=buffer_size,
    batch_size=batch_size,
    gamma=gamma,
    exploration_fraction=expl_fract,
    exploration_initial_eps=expl_init,
    exploration_final_eps=expl_fin,
    replay_buffer_class=PrioritizedReplayBuffer,
    learning_starts=learning_starts,
    gradient_steps=gradient_steps,
    train_freq=train_freq,
    target_update_interval=target_update_interval,
    tau=tau
    )

model.set_logger(new_logger)

model.learn(total_timesteps=timesteps, log_interval=4000,
            tb_log_name="DDQN_Minesweeper")

eval_env = make_vec_env(make_env, n_envs=1)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000, deterministic=True)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

model_location = "models/DDQN_agent_0.9"
model.save(model_location)
print(f"Saved model at {model_location}")

#TODO
#1. Recalibrate reward structure and hyperparameters depending on results
#3. Update readme at a later point, get a requirements.txt file
