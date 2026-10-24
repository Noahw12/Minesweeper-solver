import tkinter as tk

import gymnasium

import minesweeper as ms
#Importing it also registers the env ids with gymnasium
import minesweeper_env
from rainbow import Rainbow

difficulty = 'expert'
#Milliseconds between moves
pause = 50

#Classic Minesweeper colours for the numbers 1-8
NUMBER_COLORS = {1: '#0000ff', 2: '#008000', 3: '#ff0000', 4: '#000080',
                 5: '#800000', 6: '#008080', 7: '#000000', 8: '#808080'}
COVERED_BG, REVEALED_BG, EXPLODED_BG = '#c0c0c0', '#e0e0e0', '#ff4040'
MINE = '●'

ENV_IDS = {'beginner': 'MinesweeperBeginner-v0',
           'intermediate': 'MinesweeperIntermediate-v0',
           'expert': 'MinesweeperExpert-v0'}


class MinesweeperView:
    """tkinter view of a Board."""
    def __init__(self, board, title="Minesweeper", tile_size=32):
        self.board = board
        self.tile_size = tile_size
        self.root = tk.Tk()
        self.root.title(title)
        self.create_grid_frame()

    def create_grid_frame(self):
        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.pack(padx=5, pady=5)
        #An empty image makes the label size count in pixels. Zero padding keeps tiles the same size
        #with or without text, and a negative font size is in pixels so it ignores display scaling
        self.pixel = tk.PhotoImage(width=1, height=1)
        self.labels = {}
        for i in range(self.board.rows):
            for j in range(self.board.cols):
                lbl = tk.Label(self.grid_frame, image=self.pixel, compound='center',
                               width=self.tile_size, height=self.tile_size, bd=1, padx=0, pady=0, relief='raised',
                               bg=COVERED_BG, font=('Helvetica', -(self.tile_size * 3 // 5), 'bold'))
                lbl.grid(row=i, column=j)
                self.labels[(i, j)] = lbl

    def draw_tile(self, lbl, tile, reveal_mines=False):
        value = tile.get_value()
        if reveal_mines and tile.is_mine():
            lbl.config(text=MINE, fg='black', bg=REVEALED_BG, relief='sunken')
        elif value == ms.COVERED:
            lbl.config(text='', bg=COVERED_BG, relief='raised')
        elif value == ms.MINE:
            lbl.config(text=MINE, fg='black', bg=EXPLODED_BG, relief='sunken')
        else:
            lbl.config(text=str(value) if value else '', fg=NUMBER_COLORS.get(value, 'black'),
                       bg=REVEALED_BG, relief='sunken')

    def update_grid_frame(self, reveal_mines=False):
        for i, row in enumerate(self.board.tiles):
            for j, tile in enumerate(row):
                self.draw_tile(self.labels[(i, j)], tile, reveal_mines)

    def run(self):
        self.root.mainloop()


class AgentWatcher(MinesweeperView):
    """Plays game after game with the agent's greedy policy, with a win/loss counter."""
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
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN,
                                   anchor=tk.W, font=("Helvetica", 14))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_status_bar(self):
        self.status_var.set(f"Wins: {self.wins} | Losses: {self.losses} | Moves: {self.n_moves}")

    def reset_env_and_play(self):
        self.env.reset()
        #reset() builds a new Board object
        self.board = self.env.unwrapped.board
        self.n_moves = 0
        self.update_grid_frame()
        self.root.after(pause * 2, self.agent_play)

    def agent_play(self):
        obs = self.env.unwrapped.get_state()
        action, _ = self.model.predict(obs, deterministic=True)
        _, _, terminated, truncated, _ = self.env.step(action)
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
            self.losses += 1
            if status is ms.BoardState.LOST:
                #Show the mine that was hit
                row, col = divmod(int(action), self.board.cols)
                self.board.get_tile(row, col).set_to_true_value()
            self.root.after(pause, self.update_grid_frame)
        self.update_status_bar()
        self.root.after(pause * 2, self.reset_env_and_play)


if __name__ == "__main__":
    env = gymnasium.make(ENV_IDS[difficulty])
    model = Rainbow.load(f"models/{difficulty}/{difficulty}_model")
    AgentWatcher(env, model).run()
