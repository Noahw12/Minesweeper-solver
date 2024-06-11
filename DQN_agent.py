import gymnasium
import Minesweeper_env
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

env = gymnasium.make('Minesweeper-v0')

model = DQN("MlpPolicy", env, verbose=1, buffer_size=10000, learning_rate=1e-4, 
            batch_size=32, gamma=0.99, exploration_fraction=0.1, 
            exploration_initial_eps=1.0, exploration_final_eps=0.1)

eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)
model.learn(total_timesteps=1000, callback=eval_callback)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

model.save("dqn_minesweeper")