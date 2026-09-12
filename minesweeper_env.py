import gymnasium as gym
import numpy as np
import minesweeper as ms
from gymnasium.envs.registration import register

MINE_REWARD = -0.9
WIN_REWARD = 1.0
CLEAR_REWARD_BUDGET = 1.0
GUESS_REWARD = -0.1

class MinesweeperEnv(gym.Env):
    metadata = {'render_modes': ['ansi']}

    def __init__(self, difficulty=ms.Difficulty.BEGINNER):
        super().__init__()
        self.difficulty = difficulty
        self.board = ms.Board(difficulty, rng=self.np_random)
        self.action_space = gym.spaces.Discrete(self.board.rows * self.board.cols)
        #Values 0-8 are revealed tiles, 9 is a covered tile
        self.observation_space = gym.spaces.Box(low=0, high=ms.COVERED,
                                                shape=(self.board.rows, self.board.cols),
                                                dtype=np.int32)
        #Derived from the board so the reward scale is the same on every difficulty
        self.safe_tiles = self.board.rows * self.board.cols - self.board.mines
        self.clear_reward = CLEAR_REWARD_BUDGET / self.safe_tiles

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = ms.Board(self.difficulty, rng=self.np_random)
        return self.get_state(), {'result': self.board.game_status.name}

    #action = row * cols + col
    def step(self, action):
        row, col = divmod(int(action), self.board.cols)

        guess = self.is_guess(row, col)
        revealed = self.board.clear_tile(row, col)

        reward, terminated = self.calculate_reward(guess, revealed)
        info = {'result': self.board.game_status.name,
                'is_success': self.board.game_status is ms.BoardState.WON,
                'guess': guess,
                'revealed': revealed}

        return self.get_state(), reward, terminated, False, info

    def get_state(self):
        return self.board.get_tile_values()

    #A tile with no revealed neighbour carries no information, clearing it is a guess
    def is_guess(self, row, col):
        return all(self.board.get_tile(r, c).is_covered()
                   for r, c in self.board.neighbours(row, col))

    def calculate_reward(self, guess, revealed):

        reward = GUESS_REWARD if guess else 0.0

        if self.board.game_status is ms.BoardState.LOST:
            return reward + MINE_REWARD, True

        if self.board.game_status is ms.BoardState.WON:
            return WIN_REWARD, True

        #clear_tile returns 0 for a refused click, so only progress pays
        return reward + self.clear_reward * revealed, False

    def render(self):
        self.board.nice_print()

#A board never needs more clicks than it has safe tiles, so that is the limit
DIFFICULTY_IDS = {
    'Minesweeper-v0': ms.Difficulty.BEGINNER,
    'MinesweeperBeginner-v0': ms.Difficulty.BEGINNER,
    'MinesweeperIntermediate-v0': ms.Difficulty.INTERMEDIATE,
    'MinesweeperExpert-v0': ms.Difficulty.EXPERT,
}

for env_id, env_difficulty in DIFFICULTY_IDS.items():
    env_rows, env_cols, env_mines = ms.Board.get_dimensions(env_difficulty)
    register(
        id=env_id,
        entry_point='minesweeper_env:MinesweeperEnv',
        max_episode_steps=env_rows * env_cols - env_mines,
        kwargs={'difficulty': env_difficulty},
    )


#    MINE_REWARD = -1.0
#    WIN_REWARD = 5.0
#    SAFE_REWARD = 0.1           #flat per move, added on top of the win bonus
#    GUESS_REWARD = -SAFE_REWARD #so a blind but safe click scored exactly zero
#
#    reward = GUESS_REWARD if guess else 0.0
#    if LOST:          return reward + MINE_REWARD, True
#    if revealed > 0:  reward += SAFE_REWARD
#    if WON:           return reward + WIN_REWARD, True
#    return reward, False

