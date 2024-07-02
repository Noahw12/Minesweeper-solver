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
        self.observation_space = gym.spaces.Box(low=-1, high=10, 
                                                shape=(self.board.cols, self.board.rows), dtype=np.float32)
        self.state = self.get_state()
        
    def seed(self, seed=None):
        #Seed the random number generator if your environment uses one
        self.rng, seed = gym.utils.seeding.np_random(seed)
        return [seed]  #Return the list of seeds used in your environment

    def reset(self, seed=None, options=None):
        print(self.board.game_status)
        self.seed(seed)
        self.board = ms.Board(diff) #Reinitialize the board
        self.state = self.get_state()
        info = {'info_key': 'value'}  #Add any relevant info to track  
        return self.state, info

    #action_index = row * grid_width + col 
    #Action is encoded by the formula above
    def step(self, action):
        
        row = action // self.board.cols
        col = action % self.board.cols

        before = self.get_state()
        self.board.clearTile(row, col)
        after = self.get_state()

        reward, done = self.calculate_reward(before, after)
        self.state = self.get_state()
            
        return self.state, reward, done, False, {}

    def get_state(self):
        #Create a numerical representation of the board state
        return self.board.getTileValues()

    #Does not take into account amount of tiles cleared
    def calculate_reward(self, before, after):
        reward = 0
        done = False

        if self.board.game_status == ms.BoardState.LOST:
            #The agent hit a mine
            reward = -1.0
            done = True
        elif self.board.game_status == ms.BoardState.WON:
            #All non-mine tiles were cleared
            reward = 1.0
            done = True
        else:
            cleared_tiles = np.count_nonzero(before == 9) - np.count_nonzero(after == 9)
            reward += cleared_tiles * 0.1

        return reward, done

    def render(self):
        self.board.nicePrint() 
        
class MaskedMinesweeperEnv(gym.Wrapper):
    def __init__(self, env):
        super(MaskedMinesweeperEnv, self).__init__(env)
        self.observation_space = gym.spaces.Dict({
            "obs": env.observation_space,
            "action_mask": gym.spaces.Box(low=0, high=1, shape=(self.env.action_space.n,), dtype=bool)
        })
        
    def get_valid_actions(self):
        valid_positions = np.argwhere(self.env.unwrapped.board.getTileValues() == 9)
        valid_actions = valid_positions[:, 0] * self.env.unwrapped.board.cols + valid_positions[:, 1]
        return valid_actions
    
    def get_action_mask(self):
        mask = np.zeros(self.env.unwrapped.action_space.n, dtype=bool)
        valid_actions = self.get_valid_actions()
        mask[valid_actions] = True
        return mask
        
    def step(self, action):      
        obs, reward, done, info, truncated = self.env.unwrapped.step(action)
        action_mask = self.get_action_mask()
        return {"obs": obs, "action_mask": action_mask}, reward, done, info, truncated
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.unwrapped.reset()
        action_mask = self.get_action_mask()
        return {"obs": obs, "action_mask":action_mask}, info
    
register(
    id='Minesweeper-v0',
    entry_point='Minesweeper_env:MinesweeperEnv',
    max_episode_steps=54,
)
