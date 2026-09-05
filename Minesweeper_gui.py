import tkinter as tk
import Minesweeper as ms

#Reusable tkinter view of a Minesweeper Board. Minesweeper_watch builds its agent
#viewer on top of this, so the drawing code only lives here.
class MinesweeperView:
    def __init__(self, board, title="Minesweeper", tile_size=32):
        self.board = board
        self.tile_size = tile_size
        self.root = tk.Tk()
        self.root.title(title)

        #Load Images
        self.images = self.load_images()

        self.create_grid_frame()

    def load_images(self):
        images = {}
        try:
            images['BlankTile'] = tk.PhotoImage(file="img/BlankTile.png")
            images['ExpBomb'] = tk.PhotoImage(file="img/ExplodedBomb.png")
            images['Bomb'] = tk.PhotoImage(file="img/bomb.png")
            images['Flag'] = tk.PhotoImage(file="img/flag.png")
            #Load number images 0-8
            for i in range(0, 9):
                images[str(i) + "Tile"] = tk.PhotoImage(file=f"img/{i}Tile.png")
            return images
        except Exception as e:
            #Returning here would leave the caller with a half built window, so
            #tear the window down and let the error surface
            self.root.destroy()
            raise RuntimeError(f"An error occurred while loading images: {e}") from e

    #Point the view at another board, a gym reset() builds a new Board object
    def set_board(self, board):
        self.board = board

    def create_grid_frame(self):
        #Create a Frame to hold the grid of cells
        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.pack(padx=5, pady=5)  # Add padding around the grid

        #Init a dict to store Label widgets with (row, col) as keys
        self.labels = {}

        for i in range(self.board.rows):
            for j in range(self.board.cols):
                #Create a Label widget for each cell
                lbl = tk.Label(
                    self.grid_frame,
                    image=self.images['BlankTile'],  # Set the initial "covered" image
                    width=self.tile_size,            # Set width to match image size
                    height=self.tile_size,           # Set height to match image size
                    bd=1,                            # Border width for visual separation
                    relief="raised"                  # Raised relief to indicate covered state
                )

                #Place the Label in the grid at the specified row and column
                lbl.grid(row=i, column=j)

                #Store the Label in the labels dictionary with (i, j) as the key
                self.labels[(i, j)] = lbl

    #Draw a single tile, reveal_mines shows every mine of a finished board
    def draw_tile(self, lbl, tile, reveal_mines=False):
        value = tile.get_value()
        if reveal_mines and tile.is_mine():
            lbl.config(image=self.images['Bomb'], relief='sunken')
        elif value == ms.COVERED:
            lbl.config(image=self.images['BlankTile'], relief='raised')
        elif value == ms.FLAGGED:
            lbl.config(image=self.images['Flag'], relief='raised')
        elif value == ms.MINE:
            lbl.config(image=self.images['ExpBomb'], relief='sunken')
        else:
            lbl.config(image=self.images[f"{value}Tile"], relief='sunken')

    def update_grid_frame(self, reveal_mines=False):
        #Update the grid with the current board state
        for i, row in enumerate(self.board.tiles):
            for j, tile in enumerate(row):
                #Get the corresponding Label widget for the cell
                lbl = self.labels.get((i, j))
                self.draw_tile(lbl, tile, reveal_mines)

    def reset_covered(self):
        for lbl in self.labels.values():
            lbl.config(image=self.images['BlankTile'], relief='raised')

    def run(self):
        self.root.mainloop()

#Show a single freshly opened board
if __name__ == "__main__":
    view = MinesweeperView(ms.Board(ms.Difficulty.BEGINNER))
    view.update_grid_frame()
    view.run()
