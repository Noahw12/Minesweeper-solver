import tkinter as tk
from tkinter import messagebox

class MineSweeperGUI:
    def __init__(self, board):
        self.board = board
        self.root = tk.Tk()
        self.root.title("Minesweeper")
        
        # Load Images
        self.images = self.load_images()
        
        self.create_grid_frame()
        
        # Start the GUI loop
        self.update_grid_frame()
        self.root.mainloop()

    def load_images(self):
        images = {}
        try:
            images['BlankTile'] = tk.PhotoImage(file="img/BlankTile.png")
            images['ExpBomb'] = tk.PhotoImage(file="img/ExplodedBomb.png")
            images['Bomb'] = tk.PhotoImage(file="img/bomb.png")
            #Load number images 0-8
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

        num_rows, num_cols = self.board.shape

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
   
        for i, row in enumerate(self.board):
            for j, val in enumerate(row):
                
                # Get the corresponding Label widget for the cell
                lbl = self.labels.get((i, j))
                
                # Determine and set the appropriate image based on the cell's state
                if val == 9:
                    lbl.config(image=self.images['BlankTile'],  relief='raised')
                elif val == -1:
                    lbl.config(image=self.images['ExpBomb'], relief='sunken')
                else:
                    lbl.config(image=self.images[f"{val}Tile"], relief='sunken')