import gymnasium
from ddqn import DDQN, PrioritizedReplayBuffer, MinesweeperResCNN, MinesweeperPolicy
#Unused, but importing it registers the env ids with gymnasium
from minesweeper_env import MinesweeperEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

env_id = 'Minesweeper-v0'

def lr_schedule(progress_remaining):
    return 0.0001 + (0.0005 - 0.0001) * progress_remaining

def make_env():
    env = gymnasium.make(env_id)
    return env

buffer_size = 200000
batch_size = 64
#Horizon 1/(1-gamma) = 33 moves, against won games of about 24
gamma = 0.97
expl_fract = 0.2
expl_init = 1.0
expl_fin = 0.01
alpha = 0.6
beta_initial = 0.4

timesteps = 6000000 + buffer_size
n_envs = 16
lr = lr_schedule
#Soft updates need this every step; SB3's default of 10000 would freeze the target
target_update_interval = 1
train_freq = (1, "step")
gradient_steps = 2
learning_starts = 25000

tau_per_gradient_step = 0.000625
tau = tau_per_gradient_step * gradient_steps

vec_env = make_vec_env(make_env, n_envs=n_envs)

new_logger = configure("./logs/DDQN_Minesweeper/", ["stdout", "tensorboard"])

model = DDQN(
    MinesweeperPolicy,
    vec_env,
    policy_kwargs={
    'features_extractor_class': MinesweeperResCNN,
    'features_extractor_kwargs': {'n_filters': 128, 'n_blocks': 3},
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

model.learn(total_timesteps=timesteps, log_interval=4000,
            tb_log_name="DDQN_Minesweeper")

eval_env = make_vec_env(make_env, n_envs=1)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000, deterministic=True)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

model_location = "models/DDQN_agent_0.9"
model.save(model_location)
print(f"Saved model at {model_location}")

#TODO
#1. Reward structure
#2. tune parameters
#3. clean up code (action masking redundancy)
#5. Log the win rate, mean reward alone hides how many boards are actually cleared
#6. Update readme at a later point
