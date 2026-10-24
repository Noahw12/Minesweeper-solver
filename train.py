import os

import gymnasium
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

#Importing it also registers the env ids with gymnasium
import minesweeper_env
from networks import MinesweeperResCNN, MinesweeperPolicy
from rainbow import Rainbow
from replay_buffer import PrioritizedReplayBuffer
from utils import Checkpoint, GreedyEval, linear_schedule, save_hyperparameters

DIFFICULTIES = {
    #name            env id                          filters blocks   radius  needs   params
    'beginner':     ('MinesweeperBeginner-v0',         128,     3),   #    7      7     0.90M
    'intermediate': ('MinesweeperIntermediate-v0',     128,     7),   #   15     15     2.08M
    'expert':       ('MinesweeperExpert-v0',           128,    14),   #   29     29     4.15M
}

difficulty = 'beginner'
version = '1.0'
seed = 0
#True carries a killed run on from the newest checkpoint
resume = False

env_id, n_filters, n_blocks = DIFFICULTIES[difficulty]
model_location = f"models/{difficulty}/rainbow_{difficulty}_{version}"
log_dir = f"./logs/{difficulty}/v{version}/"
checkpoint_dir = f"models/{difficulty}/checkpoints_v{version}"

lr_initial = 0.00025
lr_final = 0.00005
buffer_size = 1000000
batch_size = 32
gamma = 0.99
expl_fract = 0.1
expl_init = 1.0
expl_fin = 0.001
learning_starts = buffer_size // 10

timesteps = 2000000
n_envs = 16
train_freq = (1, "step")
gradient_steps = 4

tau = 1.0
target_update_interval = 10000

augment_symmetry = True
#Weight of the penalty on Q-values that change when the board is turned or mirrored
symmetry_loss_coef = 1.0
#Multi-step returns, 1 is one-step DQN
n_steps = 1
#Distributional RL (C51), 1 atom turns it off
n_atoms = 51
v_min = -1.2
v_max = 2.0
dueling = True
#Noisy nets explore instead of random moves
noisy = True
if noisy:
    expl_init = expl_fin = 0.0

stats_window_size = 1000
log_interval = 250
eval_freq = 50000
eval_games = 2000

if not resume:
    save_hyperparameters(model_location + "_hyperparameters.json", {
        'difficulty': difficulty, 'version': version, 'seed': seed, 'env_id': env_id,
        'network': 'MinesweeperResCNN', 'n_filters': n_filters, 'n_blocks': n_blocks,
        'timesteps': timesteps, 'n_envs': n_envs, 'buffer_size': buffer_size,
        'learning_starts': learning_starts, 'batch_size': batch_size,
        'train_freq': list(train_freq), 'gradient_steps': gradient_steps,
        'lr_initial': lr_initial, 'lr_final': lr_final, 'gamma': gamma,
        'exploration_fraction': expl_fract, 'exploration_initial_eps': expl_init,
        'exploration_final_eps': expl_fin, 'tau': tau,
        'target_update_interval': target_update_interval,
        'augment_symmetry': augment_symmetry, 'symmetry_loss_coef': symmetry_loss_coef,
        'n_steps': n_steps, 'n_atoms': n_atoms, 'v_min': v_min, 'v_max': v_max,
        'dueling': dueling, 'noisy': noisy,
        'rewards': {'mine': minesweeper_env.MINE_REWARD, 'win': minesweeper_env.WIN_REWARD,
                    'clear_budget': minesweeper_env.CLEAR_REWARD_BUDGET,
                    'guess': minesweeper_env.GUESS_REWARD},
    }, model_location)

def make_env():
    return gymnasium.make(env_id)

vec_env = make_vec_env(make_env, n_envs=n_envs, seed=seed)

checkpoint = Checkpoint(
    #Counted in vectorised env steps, each n_envs transitions
    save_freq=max(250000 // n_envs, 1),
    save_path=checkpoint_dir,
    verbose=1,
)

greedy_eval = GreedyEval(
    difficulty=minesweeper_env.DIFFICULTY_IDS[env_id],
    eval_freq=eval_freq,
    n_games=eval_games,
    #The first boards of the fixed evaluation set
    board_seed=1_000_000,
    csv_path=model_location + "_eval_curve.csv",
    best_path=model_location + "_best",
)

if resume:
    resume_model, resume_buffer = checkpoint.paths(checkpoint.latest())
    model = Rainbow.load(resume_model, env=vec_env)
    model.load_replay_buffer(resume_buffer)
    print(f"Resuming from {resume_model} at {model.num_timesteps:,} steps")
else:
    model = Rainbow(
        MinesweeperPolicy,
        vec_env,
        policy_kwargs={
            'features_extractor_class': MinesweeperResCNN,
            'features_extractor_kwargs': {'n_filters': n_filters, 'n_blocks': n_blocks,
                                          'n_atoms': n_atoms, 'v_min': v_min, 'v_max': v_max,
                                          'dueling': dueling, 'noisy': noisy},
            'normalize_images': False,
        },
        learning_rate=linear_schedule(lr_initial, lr_final),
        verbose=1,
        tensorboard_log=log_dir,
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
        tau=tau,
        seed=seed,
        augment_symmetry=augment_symmetry,
        symmetry_loss_coef=symmetry_loss_coef,
        n_steps=n_steps,
        stats_window_size=stats_window_size,
    )

model.set_logger(configure(log_dir, ["stdout", "tensorboard"]))

#On a resume SB3 adds the requested steps to those already done
remaining = timesteps - model.num_timesteps if resume else timesteps

model.learn(total_timesteps=remaining, log_interval=log_interval,
            tb_log_name=f"{difficulty}_v{version}",
            callback=CallbackList([checkpoint, greedy_eval]),
            reset_num_timesteps=not resume)

os.makedirs(os.path.dirname(model_location), exist_ok=True)
model.save(model_location)
print(f"Saved model at {model_location}")
