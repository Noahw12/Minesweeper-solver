import gymnasium
from ddqn import DDQN, PrioritizedReplayBuffer, MinesweeperResCNN, MinesweeperPolicy
#Unused, but importing it registers the env ids with gymnasium
from minesweeper_env import MinesweeperEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
import time
import os

DIFFICULTIES = {
    #name            env id                          filters blocks   radius  needs   params   10M run
    'beginner':     ('MinesweeperBeginner-v0',         128,     3),   #    7      7     0.90M    7.5h
    'intermediate': ('MinesweeperIntermediate-v0',     128,     7),   #   15     15     2.08M   24.3h
    'expert':       ('MinesweeperExpert-v0',           128,    14),   #   29     29     4.15M   75.0h
}

difficulty = 'beginner'
version = '1.3'

env_id, n_filters, n_blocks = DIFFICULTIES[difficulty]
model_location = f"models/{difficulty}/DDQN_agent_{difficulty}_{version}"
log_dir = f"./logs/{difficulty}/v{version}/"
checkpoint_dir = f"models/{difficulty}/checkpoints_v{version}"

#True carries a killed run on from the checkpoint in checkpoint_dir, weights and
#replay buffer together. False starts fresh and overwrites whatever is there.
resume = False

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

timesteps = 10000000
n_envs = 16
train_freq = (1, "step")
gradient_steps = 4

tau = 1.0
target_update_interval = 10000

augment_symmetry = True

vec_env = make_vec_env(make_env, n_envs=n_envs)

new_logger = configure(log_dir, ["stdout", "tensorboard"])

class Checkpoint(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.model_path = os.path.join(save_path, "latest.zip")
        self.buffer_path = os.path.join(save_path, "latest_replay_buffer.pkl")

    def _init_callback(self):
        os.makedirs(self.save_path, exist_ok=True)

    @staticmethod
    def _replace(src, dst, attempts=60, delay=0.5):
        #Windows virus scanner check
        for attempt in range(attempts):
            try:
                os.replace(src, dst)
                return
            except PermissionError:
                if attempt == attempts - 1:
                    raise
                time.sleep(delay)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.model_path + ".tmp")
            self.model.save_replay_buffer(self.buffer_path + ".tmp")
            self._replace(self.model_path + ".tmp", self.model_path)
            self._replace(self.buffer_path + ".tmp", self.buffer_path)
            if self.verbose:
                print(f"Checkpoint at {self.num_timesteps:,} steps -> {self.model_path}")
        return True

checkpoint = Checkpoint(
    #Counted in calls to the vectorised env, each of which is n_envs transitions
    save_freq=max(1000000 // n_envs, 1),
    save_path=checkpoint_dir,
    verbose=1,
    )

if resume:
    model = DDQN.load(checkpoint.model_path, env=vec_env)
    model.load_replay_buffer(checkpoint.buffer_path)
    print(f"Resuming at {model.num_timesteps:,} steps with its own replay buffer")
else:
    model = DDQN(
        MinesweeperPolicy,
        vec_env,
        policy_kwargs={
        'features_extractor_class': MinesweeperResCNN,
        'features_extractor_kwargs': {'n_filters': n_filters, 'n_blocks': n_blocks},
        'normalize_images': False,
        },
        learning_rate=lr_schedule,
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
        augment_symmetry=augment_symmetry
        )

model.set_logger(new_logger)

#On a resume SB3 adds what you ask for to the steps already done
remaining = timesteps - model.num_timesteps if resume else timesteps

model.learn(total_timesteps=remaining, log_interval=4000,
            tb_log_name=f"{difficulty}_v{version}",
            callback=checkpoint,
            reset_num_timesteps=not resume)

eval_env = make_vec_env(make_env, n_envs=1)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000, deterministic=True)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

os.makedirs(os.path.dirname(model_location), exist_ok=True)
model.save(model_location)
print(f"Saved model at {model_location}")

#TODO
#1. Recalibrate reward structure and hyperparameters depending on results
#3. Update readme at a later point, get a requirements.txt file
