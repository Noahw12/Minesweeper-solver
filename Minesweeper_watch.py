import tkinter as tk
import gymnasium
import Minesweeper as ms
from DDQN import DDQN
#Importing the module is what registers the Minesweeper env ids with gymnasium
from Minesweeper_env import MinesweeperEnv
from Minesweeper_gui import MinesweeperView

#Pause between agent actions
pause = 500

class AgentWatcher(MinesweeperView):
    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.wins = 0
        self.losses = 0
        self.n_moves = 0

        super().__init__(env.unwrapped.board, title="Minesweeper RL Agent")

        self.create_status_bar()

        self.reset_env_and_play()

    def create_status_bar(self):

        self.status_var = tk.StringVar()
        self.update_status_bar()

        status_font = ("Helvetica", 14)

        self.status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=status_font
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_status_bar(self):
        self.status_var.set(f"Wins: {self.wins} | Losses: {self.losses} | Moves: {self.n_moves}")

    def reset_env_and_play(self):
        self.env.reset()
        #reset() builds a new Board, point the view at it
        self.set_board(self.env.unwrapped.board)
        self.n_moves = 0

        self.reset_covered()

        self.root.after(pause, self.update_grid_frame)

        self.root.after(pause * 2, self.agent_play)

    def agent_play(self):
        obs = self.env.unwrapped.get_state()

        #Watching the agent guess at random is not interesting, so act greedily
        action, _ = self.model.predict(obs, deterministic=True)

        _, reward, terminated, truncated, info = self.env.step(action)
        self.n_moves += 1

        self.update_status_bar()
        self.update_grid_frame()

        if terminated or truncated:
            self.end_game(action)
        else:
            self.root.after(pause, self.agent_play)

    def end_game(self, action):
        status = self.board.game_status

        if status is ms.BoardState.WON:
            self.wins += 1
            self.root.after(pause, lambda: self.update_grid_frame(reveal_mines=True))
        else:
            #A truncation is not the agent's fault, but it is not a win either
            self.losses += 1
            if status is ms.BoardState.LOST:
                row, col = divmod(int(action), self.board.cols)
                self.board.get_tile(row, col).set_to_true_value()
            self.root.after(pause, self.update_grid_frame)

        self.update_status_bar()
        self.root.after(pause * 2, self.reset_env_and_play)

if __name__ == "__main__":
    #Init environment, gymnasium.make adds the episode step limit
    env = gymnasium.make('Minesweeper-v0')

    model = DDQN.load("models/DDQN_agent_0.5")

    gui = AgentWatcher(env, model)
    gui.run()