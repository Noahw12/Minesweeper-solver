import json
import os
import re
import time

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

import minesweeper as ms


def linear_schedule(initial, final):
    def schedule(progress_remaining):
        return final + (initial - final) * progress_remaining
    return schedule


def greedy_win_rate(model, difficulty, n_games, board_seed, batch_size=1000):
    """Win rate of the noise-free policy on the boards seeded board_seed + i, playing a
    batch of games together with one network call per move."""
    policy = model.policy
    was_training = policy.training
    policy.set_training_mode(False)
    wins = 0
    for start in range(0, n_games, batch_size):
        boards = [ms.Board(difficulty, rng=np.random.default_rng(board_seed + i))
                  for i in range(start, min(start + batch_size, n_games))]
        active = boards
        while active:
            obs = th.as_tensor(np.stack([b.get_tile_values() for b in active]), device=model.device)
            with th.no_grad():
                q_values = model.apply_action_mask(policy.q_net(obs), model.action_mask(obs))
            for board, action in zip(active, q_values.argmax(dim=1).tolist()):
                board.clear_tile(*divmod(action, board.cols))
            active = [b for b in active if b.game_status is ms.BoardState.ONGOING]
        wins += sum(b.game_status is ms.BoardState.WON for b in boards)
    policy.set_training_mode(was_training)
    return wins / n_games


def save_hyperparameters(path, hyperparameters, model_location):
    """Writes the run's settings next to the model, refusing to overwrite a run."""
    if os.path.exists(model_location) or os.path.exists(path):
        raise FileExistsError(f"{model_location} already exists, bump the version")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(hyperparameters, f, indent=2)


class Checkpoint(BaseCallback):
    """Keeps only the newest model and replay buffer, named by step, for resuming."""
    def __init__(self, save_freq, save_path, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def paths(self, steps):
        return (os.path.join(self.save_path, f"model_{steps}_steps.zip"),
                os.path.join(self.save_path, f"replay_buffer_{steps}_steps.pkl"))

    def latest(self):
        """Newest step with both a model and its replay buffer."""
        names = os.listdir(self.save_path) if os.path.isdir(self.save_path) else []
        steps = [int(m.group(1)) for m in (re.fullmatch(r"replay_buffer_(\d+)_steps\.pkl", n)
                                           for n in names) if m]
        steps = [s for s in steps if os.path.exists(self.paths(s)[0])]
        if not steps:
            raise FileNotFoundError(f"No model with a matching replay buffer in {self.save_path}")
        return max(steps)

    def _init_callback(self):
        os.makedirs(self.save_path, exist_ok=True)

    @staticmethod
    def _replace(src, dst, attempts=60, delay=0.5):
        #Windows virus scanners can briefly lock a freshly written file
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
            model_path, buffer_path = self.paths(self.num_timesteps)
            self.model.save(model_path + ".tmp")
            self._replace(model_path + ".tmp", model_path)
            self.model.save_replay_buffer(buffer_path + ".tmp")
            self._replace(buffer_path + ".tmp", buffer_path)
            for name in os.listdir(self.save_path):
                old = os.path.join(self.save_path, name)
                if (re.fullmatch(r"(model|replay_buffer)_\d+_steps\.(zip|pkl)", name)
                        and old not in (model_path, buffer_path)):
                    try:
                        os.remove(old)
                    except OSError:
                        pass
            if self.verbose:
                print(f"Checkpoint at {self.num_timesteps:,} steps -> {model_path}")
        return True


class GreedyEval(BaseCallback):
    """Every eval_freq steps, plays the same n_games boards with the noise-free policy,
    logs the win rate, appends it to csv_path and saves a new best model to best_path."""
    def __init__(self, difficulty, eval_freq, n_games, board_seed, csv_path, best_path=None, verbose=1):
        super().__init__(verbose)
        self.difficulty = difficulty
        self.eval_freq = eval_freq
        self.n_games = n_games
        self.board_seed = board_seed
        self.csv_path = csv_path
        self.best_path = best_path

    def _init_callback(self):
        #self.num_timesteps is only updated once steps are taken, so read the model's
        steps_done = self.model.num_timesteps
        self.next_eval = (steps_done // self.eval_freq + 1) * self.eval_freq
        #On a resume, rows past the checkpoint belong to lost training and are redone
        rows = []
        if os.path.exists(self.csv_path):
            with open(self.csv_path) as f:
                rows = [row for row in f.read().splitlines()[1:]
                        if int(row.split(',')[0]) <= steps_done]
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, 'w') as f:
            f.write("steps,win_rate,standard_error,games\n")
            f.writelines(row + "\n" for row in rows)
        self.best_rate = max((float(row.split(',')[1]) for row in rows), default=-1.0)

    def win_rate(self):
        return greedy_win_rate(self.model, self.difficulty, self.n_games, self.board_seed)

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            rate = self.win_rate()
            se = (rate * (1 - rate) / self.n_games) ** 0.5
            self.logger.record('eval/win_rate', rate)
            self.logger.record('eval/standard_error', se)
            self.logger.dump(self.num_timesteps)
            with open(self.csv_path, 'a') as f:
                f.write(f"{self.num_timesteps},{rate:.4f},{se:.4f},{self.n_games}\n")
            if self.verbose:
                print(f"Eval at {self.num_timesteps:,} steps: win rate {rate:.4f} +/- {se:.4f}")
            if self.best_path is not None and rate > self.best_rate:
                self.best_rate = rate
                self.model.save(self.best_path)
                if self.verbose:
                    print(f"New best, saved to {self.best_path}")
            self.next_eval += self.eval_freq
        return True
