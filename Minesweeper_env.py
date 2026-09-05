import gymnasium as gym
import numpy as np
import Minesweeper as ms
from gymnasium.envs.registration import register

#Reward for stepping on a mine
MINE_REWARD = -1.0
#Bonus for clearing the whole board
WIN_REWARD = 1.0
#Penalty for picking a tile that is already revealed or flagged. Such a move does
#not change the board, so without a penalty the agent can farm reward forever.
INVALID_ACTION_REWARD = -1.0
#Reward for safely uncovering a tile. It has to stay within a small factor of
#MINE_REWARD: if it does not, the gap in value between a tile that is known to be
#safe and an average unknown tile shrinks below the error the network fits to, and
#the agent only ever learns to dodge mines instead of making progress.
#It is a flat reward per move and not per revealed tile, otherwise a lucky flood
#fill would outweigh the mine penalty and the agent turns reckless.
SAFE_REWARD = 0.3
#Penalty for clearing a tile that has no revealed neighbour. Such a move cannot be
#deduced from anything on the board, it is a blind guess. Penalising it pushes the
#agent onto the frontier, which is the only place where deductions are possible.
GUESS_REWARD = -0.3

#An opening move can, very rarely, clear the whole board. Reroll a few times so
#that the agent is never handed an already finished game.
MAX_BOARD_ATTEMPTS = 100

class MinesweeperEnv(gym.Env):
    metadata = {'render_modes': ['ansi']}

    def __init__(self, difficulty=ms.Difficulty.BEGINNER):
        super().__init__()
        self.difficulty = difficulty
        rows, cols, _ = ms.Board.get_dimensions(difficulty)
        self.board = self._new_board()
        self.action_space = gym.spaces.Discrete(rows * cols)
        #Values 0-8 are revealed tiles, 9 is a covered tile. The environment never
        #flags, so ms.FLAGGED is outside the observation space on purpose.
        self.observation_space = gym.spaces.Box(low=0, high=ms.COVERED,
                                                shape=(rows, cols), dtype=np.int32)
        self.state = self.get_state()

    def _new_board(self):
        for _ in range(MAX_BOARD_ATTEMPTS):
            #self.np_random is seeded by reset(seed=...), which is what makes
            #episodes reproducible
            board = ms.Board(self.difficulty, rng=self.np_random)
            if board.game_status is ms.BoardState.ONGOING:
                return board
        return board

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = self._new_board() #Reinitialize the board
        self.state = self.get_state()
        info = {'result': self.board.game_status.name}
        return self.state, info

    #action_index = row * grid_width + col
    #Action is encoded by the formula above
    def step(self, action):
        row, col = divmod(int(action), self.board.cols)

        #The board is authoritative about what is legal, the action mask below is
        #only an aid for the agent
        invalid = not self.board.get_tile(row, col).is_covered()
        #Has to be measured before the tile is cleared
        guess = False if invalid else self.is_guess(row, col)
        revealed = 0 if invalid else self.board.clear_tile(row, col)

        self.state = self.get_state()
        reward, terminated = self.calculate_reward(invalid, guess, revealed)
        info = {'result': self.board.game_status.name,
                'invalid_action': invalid,
                'guess': guess,
                'revealed': revealed}

        return self.state, reward, terminated, False, info

    def get_state(self):
        #Create a numerical representation of the board state
        return self.board.get_tile_values()

    #True for every tile the agent is still allowed to clear
    def action_masks(self):
        return (self.board.get_tile_values() == ms.COVERED).flatten()

    #A tile with no revealed neighbour carries no information, clearing it is a guess
    def is_guess(self, row, col):
        for dr, dc in ms.NEIGHBOURS:
            r, c = row + dr, col + dc
            if 0 <= r < self.board.rows and 0 <= c < self.board.cols:
                if not self.board.get_tile(r, c).is_covered():
                    return False
        return True

    def calculate_reward(self, invalid, guess, revealed):

        if invalid:
            #The agent picked a tile that is not covered, the board did not change
            return INVALID_ACTION_REWARD, False

        #The penalty is for the decision, so it applies whatever the move uncovered
        reward = GUESS_REWARD if guess else 0.0

        if self.board.game_status is ms.BoardState.LOST:
            #The agent hit a mine
            return reward + MINE_REWARD, True

        reward += SAFE_REWARD

        if self.board.game_status is ms.BoardState.WON:
            #All non-mine tiles were cleared
            return reward + WIN_REWARD, True

        return reward, False

    def render(self):
        self.board.nice_print()

#A board can never need more clicks than it has safe tiles, so that is the
#episode limit. Registering per difficulty keeps the limit in sync with the board.
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
