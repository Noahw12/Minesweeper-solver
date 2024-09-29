import tkinter as tk
import numpy as np
from stable_baselines3 import DQN  # Or DDQN in your case
from Minesweeper_env import MinesweeperEnv  # Import your environment
from tkinter import messagebox

class MineSweeperGUI:
    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.root = tk.Tk()
        self.root.title("Minesweeper")

        # Load Images
        self.images = self.load_images()
        
        self.create_grid_frame()
        
        # Start the agent's gameplay
        self.reset_env_and_play()

        # Start the GUI loop
        self.root.mainloop()

    def load_images(self):
        images = {}
        try:
            images['BlankTile'] = tk.PhotoImage(file="img/BlankTile.png")
            images['ExpBomb'] = tk.PhotoImage(file="img/ExplodedBomb.png")
            images['Bomb'] = tk.PhotoImage(file="img/bomb.png")
            # Load number images 0-8
            for i in range(0, 9):
                images[str(i) + "Tile"] = tk.PhotoImage(file=f"img/{i}Tile.png")
            return images
        except Exception as e:
            messagebox.showerror("Image Loading Error", f"An error occurred while loading images: {e}")
            self.root.destroy()

    def create_grid_frame(self):
        # Create a Frame to hold the grid of cells
        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.pack(padx=5, pady=5)  # Add padding around the grid

        # Initialize a dictionary to store Label widgets with their (row, col) as keys
        self.labels = {}

        num_rows, num_cols = self.env.board.rows, self.env.board.cols

        # Iterate over each row and column index to create Label widgets
        for i in range(num_rows):
            for j in range(num_cols):
                # Create a Label widget for each cell
                lbl = tk.Label(
                    self.grid_frame,
                    image=self.images['BlankTile'],  # Set the initial "covered" image
                    width=16,                        # Set width to match image size
                    height=16,                       # Set height to match image size
                    bd=1,                            # Border width for visual separation
                    relief="raised"                  # Raised relief to indicate covered state
                )

                # Place the Label in the grid at the specified row and column
                lbl.grid(row=i, column=j)

                # Store the Label in the labels dictionary with (i, j) as the key
                self.labels[(i, j)] = lbl

    def update_grid_frame(self):
        # Update the grid with the current board state
        for i, row in enumerate(self.env.board):
            for j, tile in enumerate(row):
                lbl = self.labels.get((i, j))
                value = tile.get_value()
                if value == 9:
                    lbl.config(image=self.images['BlankTile'], relief='raised')
                elif value == -1:
                    lbl.config(image=self.images['ExpBomb'], relief='sunken')
                else:
                    lbl.config(image=self.images[f"{value}Tile"], relief='sunken')

    def reset_env_and_play(self):
        # Reset the environment before starting a new episode
        self.env.reset()

        # Start the agent playing
        self.agent_play()

    def agent_play(self):
        obs = self.env.get_state()  # Get the current board state as an observation
        obs = obs.flatten()  # Flatten for the agent's input if needed

        action, _states = self.model.predict(obs)  # Agent predicts the action

        # Step in the environment with the predicted action
        new_state, reward, done, _, _ = self.env.step(action)

        # Update the GUI
        self.update_grid_frame()

        # Check if the game is done (won/lost)
        if done:
            self.end_game(reward)
        else:
            # Schedule the next move after a delay (e.g., 500 ms)
            self.root.after(500, self.agent_play)

    def end_game(self, reward):
        # Handle the game end scenario (either win or loss)
        if reward == 1.0:
            tk.messagebox.showinfo("Game Over", "You Won!")
        else:
            tk.messagebox.showinfo("Game Over", "You Lost!")
        # Reset the board and start again
        self.reset_env_and_play()

# Now initialize the environment and load the model
env = MinesweeperEnv()  # Initialize your custom Minesweeper environment

# Load your trained RL model (replace with the correct path to your model)
model = DQN.load("models/DDQN_agent_0.1")

# Start the Tkinter GUI with the environment and model
gui = MineSweeperGUI(env, model)
