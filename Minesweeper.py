import numpy as np
from collections import deque
from enum import Enum

#Tile.value encodes what a player is allowed to see
COVERED = 9
#Tile.trueValue uses -1 for a mine and 0-8 for the number of adjacent mines
MINE = -1

#The eight neighbours of a tile
NEIGHBOURS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

class Tile:
    def __init__(self):
        self.value = COVERED
        self.trueValue = np.nan

    def get_value(self):
        return self.value

    #Used for the tkinter GUI only
    def set_to_true_value(self):
        self.value = self.trueValue

    def get_true_value(self):
        return self.trueValue

    def set_true_value(self, newValue):
        self.trueValue = newValue

    def is_covered(self):
        return self.value == COVERED

    def is_mine(self):
        return self.trueValue == MINE

    def clear(self):
        if self.trueValue == MINE:
            return self.trueValue
        else:
            self.value = self.trueValue
            return self.value

class Difficulty(Enum):
    BEGINNER = 1
    INTERMEDIATE = 2
    EXPERT = 3

class BoardState(Enum):
    ONGOING = 0
    WON = 1
    LOST = 2

class Board:
    #Pass the gym environment's rng to make boards reproducible
    def __init__(self, diff, rng=None):
        self.diff = diff
        self.rng = rng if rng is not None else np.random.default_rng()
        self.rows, self.cols, self.mines = self.get_dimensions(diff)
        self.safe_tiles = self.rows * self.cols - self.mines
        self.tiles = self.create_board()
        self.game_status = BoardState.ONGOING
        #Mines are placed on the first clear, so the opening tile is always safe
        #and choosing it is a real decision
        self.mines_placed = False

    @staticmethod
    def get_dimensions(diff):
        if diff == Difficulty.BEGINNER:
            return 8, 8, 10
        elif diff == Difficulty.INTERMEDIATE:
            return 16, 16, 40
        elif diff == Difficulty.EXPERT:
            return 16, 30, 99
        raise ValueError(f"Unknown difficulty: {diff}")

    def create_board(self):
        return [[Tile() for _ in range(self.cols)] for _ in range(self.rows)]

    def _place_mines(self, safe_zone):
        all_positions = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        available_positions = [pos for pos in all_positions if pos not in safe_zone]
        if len(available_positions) < self.mines:
            raise ValueError(f"Cannot place {self.mines} mines outside a safe zone of "
                             f"{len(safe_zone)} tiles on a {self.rows}x{self.cols} board")
        order = self.rng.permutation(len(available_positions))
        for index in order[:self.mines]:
            r, c = available_positions[index]
            self.get_tile(r, c).set_true_value(MINE)

    def _count_adjacent_mines(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.get_tile(r, c).is_mine():
                    continue
                count = sum(self.get_tile(r + dr, c + dc).is_mine() for dr, dc in NEIGHBOURS
                            if 0 <= r + dr < self.rows and 0 <= c + dc < self.cols)
                self.get_tile(r, c).set_true_value(count)

    def get_tile(self, row, col):
        return self.tiles[row][col]

    def get_tile_values(self):
        return np.array([[tile.get_value() for tile in row] for row in self.tiles], dtype=np.int32)

    def get_true_tile_values(self):
        return np.array([[tile.get_true_value() for tile in row] for row in self.tiles])

    #The board is won once every tile that is still covered is a mine
    def win_check(self):
        if self.game_status is not BoardState.ONGOING:
            return
        hidden = np.count_nonzero(self.get_tile_values() == COVERED)
        if hidden == self.mines:
            self.game_status = BoardState.WON

    def nice_print(self):
        for row in self.tiles:
            niceRow = ''
            for tile in row:
                if tile.get_value() == COVERED:
                    niceRow += '?' + ' '
                elif tile.get_value() == MINE:
                    niceRow += 'B' + ' '
                else:
                    niceRow += str(tile.get_value()) + ' '
            print(niceRow)

    #Flood fill from a tile with no adjacent mines. Returns how many it revealed.
    def read_tiles(self, row, col):
        queue = deque([(row, col)])
        visited = set([(row, col)])
        revealed = 0

        while queue:
            current_row, current_col = queue.popleft()

            for dx, dy in NEIGHBOURS:
                nx, ny = current_row + dx, current_col + dy

                if not (0 <= nx < self.rows and 0 <= ny < self.cols) or (nx, ny) in visited:
                    continue
                visited.add((nx, ny))
                tile = self.get_tile(nx, ny)
                #A revealed tile was expanded already, and a fill never opens a mine
                if not tile.is_covered() or tile.is_mine():
                    continue
                if tile.clear() == 0:
                    queue.append((nx, ny))
                revealed += 1

        return revealed

    #Returns how many tiles were revealed; a mine, a finished board or an
    #already revealed tile reveals nothing
    def clear_tile(self, row, col):
        if self.game_status is not BoardState.ONGOING:
            return 0

        tile = self.get_tile(row, col)
        if not tile.is_covered():
            return 0

        #Classic first click rule: the mines avoid the opening tile, and only it
        if not self.mines_placed:
            self._place_mines({(row, col)})
            self._count_adjacent_mines()
            self.mines_placed = True

        tile_value = tile.clear()
        if tile_value == MINE:
            self.game_status = BoardState.LOST
            return 0

        revealed = 1
        if tile_value == 0:
            revealed += self.read_tiles(row, col)
        self.win_check()
        return revealed
