import gymnasium
from DDQN import ActionMaskPolicy, DDQN, PrioritizedReplayBuffer
from Minesweeper_env import MinesweeperEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

def lr_schedule(progress_remaining):
    return 0.00025 + (0.001 - 0.00025) * progress_remaining

def make_env():
    env = gymnasium.make('Minesweeper-v0')  
    return env 

buffer_size = 10000
batch_size = 32
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

model.save("models/DDQN_agent_0.1")
       
#TODO
#4. Reward structure
#5. Check for bugs
#6. tune parameters