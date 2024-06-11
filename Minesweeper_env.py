import gymnasium as gym
import numpy as np
import Minesweeper as ms
from gymnasium.envs.registration import register

class MinesweeperEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.board = ms.Board()
        self.action_space = gym.spaces.Discrete(8 * 8 * 1)
        self.observation_space = gym.spaces.Box(low=-1, high=10, shape=(8, 8), dtype=np.int32)
        self.state = self.get_state()
        
    def seed(self, seed=None):
        # Seed the random number generator if your environment uses one
        self.rng, seed = gym.utils.seeding.np_random(seed)
        return [seed]  # Return the list of seeds used in your environment

    def reset(self, seed=None, options=None):
        print(self.board.game_status)
        self.seed(seed)
        self.board.resetGame() #Reinitialize the board
        self.state = self.get_state()
        info = {'info_key': 'value'}  #Add any relevant info to track  
        return self.state, info

    #action_index = (row * grid_width + col) * num_action_types + action_type 
    #Action is encoded by the formula above
    def step(self, action):
       
        if self.board.game_status != ms.BoardState.ONGOING:
            return self.state, 0, True, False, {} #Handle odd cases where agent attemps action after ending game
        
        num_action_types = 1  #Only reveal
        grid_width = 8 #beginner mode
        
        # Decode the action
        action_type = action % num_action_types
        position = action // num_action_types
        row = position // grid_width
        col = position % grid_width
       
        if action_type == 0:
            self.board.clearTile(row, col)
        # else:
        #     self.board.flagTile(row, col)

        reward, done = self.calculate_reward()
        self.state = self.get_state()
            
        return self.state, reward, done, False, {}

    def get_state(self):
        # Create a numerical representation of the board state
        return self.board.getTileValues()

    def calculate_reward(self):
        reward = 0
        done = False

        if self.board.game_status == ms.BoardState.LOST:
            # The agent hit a mine
            reward = -1.0
            done = True
        elif self.board.game_status == ms.BoardState.WON:
            # All non-mine tiles were cleared
            reward = 1.0
            done = True
        else:
            reward = 0.1
            # Game is still ongoing
            #if self.board.last_action_cleared_safe_tile():
                # Last action successfully cleared a safe tile
                #reward += 0.1
            # if self.board.last_action_flagged_correctly():
            #     # Correctly flagged a mine
            #     reward += 0.5
            # if self.board.last_action_flagged_incorrectly():
            #     # Incorrectly flagged a tile
            #     reward -= 0.3
        return reward, done

    def render(self):
        self.board.niceprint()   
    
register(
    id='Minesweeper-v0',
    entry_point='Minesweeper_env:MinesweeperEnv',
)
