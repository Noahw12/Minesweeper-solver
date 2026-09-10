import gymnasium as gym
import numpy as np
import Minesweeper as ms
from gymnasium.envs.registration import register

MINE_REWARD = -1.0
WIN_REWARD = 5.0
#Without this the agent can farm reward on moves that change nothing
INVALID_ACTION_REWARD = -1.0
#Shaping while wins are still too rare to learn from. Flat per move, not per
#revealed tile, or a lucky flood fill would outweigh the mine penalty.
SAFE_REWARD = 0.1
#A blind guess. Tied to SAFE_REWARD so a blind but safe click is worth zero.
GUESS_REWARD = -SAFE_REWARD

class MinesweeperEnv(gym.Env):
    metadata = {'render_modes': ['ansi']}

    def __init__(self, difficulty=ms.Difficulty.BEGINNER):
        super().__init__()
        self.difficulty = difficulty
        rows, cols, _ = ms.Board.get_dimensions(difficulty)
        self.board = ms.Board(difficulty, rng=self.np_random)
        self.action_space = gym.spaces.Discrete(rows * cols)
        #Values 0-8 are revealed tiles, 9 is a covered tile
        self.observation_space = gym.spaces.Box(low=0, high=ms.COVERED,
                                                shape=(rows, cols), dtype=np.int32)
        self.state = self.get_state()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = ms.Board(self.difficulty, rng=self.np_random)
        self.state = self.get_state()
        info = {'result': self.board.game_status.name}
        return self.state, info

    #action = row * cols + col
    def step(self, action):
        row, col = divmod(int(action), self.board.cols)

        #Has to be measured before the tile is cleared
        guess = self.is_guess(row, col)
        #clear_tile refuses a tile that is not covered and returns 0, so nothing
        #forces a caller to respect a mask and legality needs no separate check
        revealed = self.board.clear_tile(row, col)

        self.state = self.get_state()
        reward, terminated = self.calculate_reward(guess, revealed)
        info = {'result': self.board.game_status.name,
                'guess': guess,
                'revealed': revealed}

        return self.state, reward, terminated, False, info

    def get_state(self):
        return self.board.get_tile_values()

    #A tile with no revealed neighbour carries no information, clearing it is a guess
    def is_guess(self, row, col):
        for dr, dc in ms.NEIGHBOURS:
            r, c = row + dr, col + dc
            if 0 <= r < self.board.rows and 0 <= c < self.board.cols:
                if not self.board.get_tile(r, c).is_covered():
                    return False
        return True

    def calculate_reward(self, guess, revealed):

        #The penalty is for the decision, so it applies whatever the move uncovered
        reward = GUESS_REWARD if guess else 0.0

        if self.board.game_status is ms.BoardState.LOST:
            return reward + MINE_REWARD, True

        #Nothing was uncovered, so the tile was already revealed
        if revealed == 0:
            return INVALID_ACTION_REWARD, False

        reward += SAFE_REWARD

        if self.board.game_status is ms.BoardState.WON:
            return reward + WIN_REWARD, True

        return reward, False

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
        entry_point='Minesweeper_env:MinesweeperEnv',
        max_episode_steps=env_rows * env_cols - env_mines,
        kwargs={'difficulty': env_difficulty},
    )
