import tkinter as tk
import minesweeper as ms

#Reusable tkinter view of a Minesweeper Board. Minesweeper_watch builds its agent
#viewer on top of this, so the drawing code only lives here.
class MinesweeperView:
    def __init__(self, board, title="Minesweeper", tile_size=32):
        self.board = board
        self.tile_size = tile_size
        self.root = tk.Tk()
        self.root.title(title)

        self.images = self.load_images()

        self.create_grid_frame()

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
            #Returning would leave the caller a half built window
            self.root.destroy()
            raise RuntimeError(f"An error occurred while loading images: {e}") from e

    #Point the view at another board, a gym reset() builds a new Board object
    def set_board(self, board):
        self.board = board

    def create_grid_frame(self):
        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.pack(padx=5, pady=5)

        self.labels = {}

        for i in range(self.board.rows):
            for j in range(self.board.cols):
                lbl = tk.Label(
                    self.grid_frame,
                    image=self.images['BlankTile'],
                    width=self.tile_size,
                    height=self.tile_size,
                    bd=1,
                    relief="raised"
                )

                lbl.grid(row=i, column=j)

                self.labels[(i, j)] = lbl

    #Draw a single tile, reveal_mines shows every mine of a finished board
    def draw_tile(self, lbl, tile, reveal_mines=False):
        value = tile.get_value()
        if reveal_mines and tile.is_mine():
            lbl.config(image=self.images['Bomb'], relief='sunken')
        elif value == ms.COVERED:
            lbl.config(image=self.images['BlankTile'], relief='raised')
        elif value == ms.MINE:
            lbl.config(image=self.images['ExpBomb'], relief='sunken')
        else:
            lbl.config(image=self.images[f"{value}Tile"], relief='sunken')

    def update_grid_frame(self, reveal_mines=False):
        for i, row in enumerate(self.board.tiles):
            for j, tile in enumerate(row):
                lbl = self.labels.get((i, j))
                self.draw_tile(lbl, tile, reveal_mines)

    def run(self):
        self.root.mainloop()

#Show a fresh board
if __name__ == "__main__":
    view = MinesweeperView(ms.Board(ms.Difficulty.BEGINNER))
    view.update_grid_frame()
    view.run()
