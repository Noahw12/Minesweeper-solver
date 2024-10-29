import tkinter as tk
import numpy as np
from DDQN import DDQN
from Minesweeper_env import MinesweeperEnv 
from tkinter import messagebox

#Pause between agent actions
pause = 500

class MineSweeperGUI:
    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.wins = 0
        self.losses = 0
        self.root = tk.Tk()
        self.root.title("Minesweeper RL Agent")

        #Load Images
        self.images = self.load_images()
        
        self.create_grid_frame()
        
        #Create win/loss counter
        self.create_status_bar()

        #Start the agent's gameplay
        self.reset_env_and_play()

        #Start the GUI loop
        self.root.mainloop()

    def load_images(self):
        images = {}
        try:
            images['BlankTile'] = tk.PhotoImage(file="img/BlankTile.png")
            images['ExpBomb'] = tk.PhotoImage(file="img/ExplodedBomb.png")
            images['Bomb'] = tk.PhotoImage(file="img/bomb.png")
            for i in range(0, 9):
                images[str(i) + "Tile"] = tk.PhotoImage(file=f"img/{i}Tile.png")
            return images
        except Exception as e:
            messagebox.showerror("Image Loading Error", f"An error occurred while loading images: {e}")
            self.root.destroy()
            
    def create_status_bar(self):
    
        self.status_var = tk.StringVar()
        self.status_var.set(f"Wins: {self.wins} | Losses: {self.losses}")
        
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
        self.status_var.set(f"Wins: {self.wins} | Losses: {self.losses}")

    def create_grid_frame(self):
        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.pack(padx=5, pady=5)  

        #Init a dict to store Label widgets with (row, col) as keys
        self.labels = {}

        num_rows, num_cols = self.env.board.rows, self.env.board.cols

        for i in range(num_rows):
            for j in range(num_cols):
                #Create a Label widget for each cell
                lbl = tk.Label(
                    self.grid_frame,
                    image=self.images['BlankTile'],  
                    width=32,                        
                    height=32,                       
                    bd=1, #Border width
                    relief="raised"
                )

                #Place the Label in the grid at row and column
                lbl.grid(row=i, column=j)
                self.labels[(i, j)] = lbl           
                    
    def update_grid_frame(self):
        #Update the grid with the current board state
        for i, row in enumerate(self.env.board.tiles):
            for j, tile in enumerate(row):
                lbl = self.labels.get((i, j))
                value = tile.get_value()
                if value == 9:
                    lbl.config(image=self.images['BlankTile'], relief='raised')
                elif value == -1:
                    lbl.config(image=self.images['ExpBomb'], relief='sunken')
                else:
                    lbl.config(image=self.images[f"{value}Tile"], relief='sunken')
                    
    def update_grid_frame_win(self):
        #Update the grid with the current board state
        for i, row in enumerate(self.env.board.tiles):
            for j, tile in enumerate(row):
                lbl = self.labels.get((i, j))
                
                if tile.get_true_value() == -1:
                    lbl.config(image=self.images['Bomb'], relief='sunken')
                else:
                    value = tile.get_value()
                    if value == 9:
                        lbl.config(image=self.images['BlankTile'], relief='raised')   
                    else:
                        lbl.config(image=self.images[f"{value}Tile"], relief='sunken')
    
    def reset_covered(self):              
        for lbl in self.labels.values():
            lbl.config(image=self.images['BlankTile'], relief='raised')

    def reset_env_and_play(self):
        #Reset the environment before starting a new game
        self.env.reset()
        
        #Update the GUI to show all tiles as covered
        self.reset_covered()
        
        #Update the GUI to show tiles after first reveal
        self.root.after(pause, self.update_grid_frame)    

        #Start the agent playing
        self.root.after(pause * 2, self.agent_play)

    def agent_play(self):
        #Get the board state as observation
        obs = self.env.get_state()  
        
        #Add batch dimension
        obs = np.expand_dims(obs, axis=0)

        action, _ = self.model.predict(obs)  
        action = action[0]
        
        #Step in the environment with the predicted action
        _, reward, done, _, _ = self.env.step(action)

        #Update the GUI
        self.update_grid_frame()

        #Check if game is done
        if done:
            self.end_game(reward, action)
        else:
            self.root.after(pause, self.agent_play)

    def end_game(self, reward, action):
        #Reveal all mines if won
        if reward == 1.0:
            self.wins += 1
            self.root.after(pause, self.update_grid_frame_win)
            self.update_status_bar()
        #Reveal clicked mine if lost
        else:
            self.losses += 1
            #Update the board with the clicked bomb
            row = action // self.env.board.cols
            col = action % self.env.board.cols
            bomb_tile = self.env.board.get_tile(row,col)
            bomb_tile.set_to_true_value()
            self.root.after(pause, self.update_grid_frame)
            self.update_status_bar()
            
        self.root.after(pause*2, self.reset_env_and_play)

#Init environment
env = MinesweeperEnv()  

#Load the trained RL model 
model = DDQN.load("models/DDQN_agent_0.1")

#Start the Tkinter GUI 
gui = MineSweeperGUI(env, model)
