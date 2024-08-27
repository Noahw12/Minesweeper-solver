import gymnasium as gym
import numpy as np
import Minesweeper as ms
from gymnasium.envs.registration import register

diff = ms.Difficulty.BEGINNER

class MinesweeperEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.board = ms.Board(diff)
        self.action_space = gym.spaces.Discrete(self.board.cols * self.board.rows)
        self.observation_space = gym.spaces.Box(low=0, high=9, 
                                                shape=(self.board.cols, self.board.rows), dtype=np.int32)
        self.state = self.get_state()
        
    def seed(self, seed=None):
        #Seed the random number generator if your environment uses one
        self.rng, seed = gym.utils.seeding.np_random(seed)
        return [seed]  #Return the list of seeds used in your environment

    def reset(self, seed=None, options=None):
        info = {'Result': self.board.game_status}  
        self.seed(seed)
        self.board = ms.Board(diff) #Reinitialize the board
        self.state = self.get_state()
        return self.state, info

    #action_index = row * grid_width + col 
    #Action is encoded by the formula above
    def step(self, action):
        
        row = action // self.board.cols
        col = action % self.board.cols

        self.board.clear_tile(row, col)
        new_state = self.get_state()

        reward, done = self.calculate_reward()
            
        return new_state, reward, done, False, {}

    def get_state(self):
        #Create a numerical representation of the board state
        return self.board.get_tile_values()

    def calculate_reward(self):

        if self.board.game_status == ms.BoardState.LOST:
            #The agent hit a mine
            reward = -1.0
            done = True
        elif self.board.game_status == ms.BoardState.WON:
            #All non-mine tiles were cleared
            reward = 1.0
            done = True
        else:
            reward = 0.1
            done = False

        return reward, done

    def render(self):
        self.board.nice_print() 

#Rows* Columns - Mines
beginner_max_steps = 8*8 - 10
intermediate_max_steps = 16*16 - 40
expert_max_steps = 16*30 - 99

register(
    id='Minesweeper-v0',
    entry_point='Minesweeper_env:MinesweeperEnv',
    max_episode_steps=beginner_max_steps,
)
