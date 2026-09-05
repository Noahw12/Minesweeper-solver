import numpy as np
from collections import deque
from enum import Enum

#Tile.value encodes what a player is allowed to see
COVERED = 9
FLAGGED = 10
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

    #Only a covered tile may be cleared, a flagged tile is protected
    def is_covered(self):
        return self.value == COVERED

    def is_flagged(self):
        return self.value == FLAGGED

    def is_mine(self):
        return self.trueValue == MINE

    def clear(self):
        if self.trueValue == MINE:
            return self.trueValue
        else:
            self.value = self.trueValue
            return self.value

    def flag(self):
        #Revealed tiles must keep their value, otherwise board state is lost
        if self.value == COVERED:
            self.value = FLAGGED
            return True
        return False

    def unflag(self):
        if self.value == FLAGGED:
            self.value = COVERED
            return True
        return False

class Difficulty(Enum):
    BEGINNER = 1
    INTERMEDIATE = 2
    EXPERT = 3

class BoardState(Enum):
    ONGOING = 0
    WON = 1
    LOST = 2

class Board:
    #rng is a numpy Generator. Pass the one owned by the gym environment to make
    #boards reproducible, without it every board comes from a fresh generator.
    def __init__(self, diff, rng=None, first_move=True):
        self.diff = diff
        self.rng = rng if rng is not None else np.random.default_rng()
        self.rows, self.cols, self.mines = self.get_dimensions(diff)
        self.safe_tiles = self.rows * self.cols - self.mines
        self.tiles = self.create_board()
        self.game_status = BoardState.ONGOING
        #Make the first move
        if first_move:
            self.first_clear(int(self.rng.integers(0, self.rows)),
                             int(self.rng.integers(0, self.cols)))

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

    #Calculate adjacent mines for every cell of the board that is not a mine
    def _count_adjacent_mines(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.get_tile(r, c).is_mine():
                    continue
                count = sum(self.get_tile(r + dr, c + dc).is_mine() for dr, dc in NEIGHBOURS
                            if 0 <= r + dr < self.rows and 0 <= c + dc < self.cols)
                self.get_tile(r, c).set_true_value(count)

    #Opening move that only guarantees the clicked tile itself to be safe
    def first_clear_small_safe_zone(self, row, col):
        self._place_mines({(row, col)})
        self._count_adjacent_mines()
        return self.clear_tile(row, col)

    #Opening move that guarantees the clicked tile and its neighbours to be safe
    def first_clear(self, row, col):
        safe_zone = {(row + dr, col + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                     if 0 <= row + dr < self.rows and 0 <= col + dc < self.cols}
        self._place_mines(safe_zone)
        self._count_adjacent_mines()
        return self.clear_tile(row, col)

    def get_tile(self, row, col):
        return self.tiles[row][col]

    def get_tile_values(self):
        return np.array([[tile.get_value() for tile in row] for row in self.tiles], dtype=np.int32)

    def get_true_tile_values(self):
        return np.array([[tile.get_true_value() for tile in row] for row in self.tiles])

    #The board is won once every tile that is still hidden is a mine. Counting
    #covered tiles alone would never trigger on a board where mines are flagged.
    def win_check(self):
        if self.game_status is not BoardState.ONGOING:
            return
        values = self.get_tile_values()
        hidden = np.count_nonzero((values == COVERED) | (values == FLAGGED))
        if hidden == self.mines:
            self.game_status = BoardState.WON

    def nice_print(self):
        for row in self.tiles:
            niceRow = ''
            for tile in row:
                if tile.get_value() == COVERED:
                    niceRow += '?' + ' '
                elif tile.get_value() == FLAGGED:
                    niceRow += 'F' + ' '
                elif tile.get_value() == MINE:
                    niceRow += 'B' + ' '
                else:
                    niceRow += str(tile.get_value()) + ' '
            print(niceRow)

    def nice_print_true_values(self):
        for row in self.tiles:
            niceRow = ''
            for tile in row:
                if tile.get_true_value() == MINE:
                    niceRow += 'B' + ' '
                else:
                    niceRow += str(tile.get_true_value()) + ' '
            print(niceRow)

    #Flood fill starting from a tile without adjacent mines.
    #Returns the number of tiles the fill revealed.
    def read_tiles(self, row, col):

        queue = deque([(row, col)])
        visited = set([(row, col)])
        revealed = 0

        while queue:
            current_row, current_col = queue.popleft()

            for dx, dy in NEIGHBOURS:
                nx, ny = current_row + dx, current_col + dy

                #Check bounds and if the tile has not been visited
                if not (0 <= nx < self.rows and 0 <= ny < self.cols) or (nx, ny) in visited:
                    continue
                visited.add((nx, ny))
                tile = self.get_tile(nx, ny)
                #An already revealed tile was expanded when it was revealed, and a
                #flagged tile or a mine is never opened by a flood fill
                if not tile.is_covered() or tile.is_mine():
                    continue
                if tile.clear() == 0:  #Process the tile
                    queue.append((nx, ny))
                revealed += 1

        return revealed

    #Reveal a tile and return how many tiles that revealed. Clearing a mine, a
    #finished board or a tile that is not covered reveals nothing.
    def clear_tile(self, row, col):
        if self.game_status is not BoardState.ONGOING:
            return 0

        tile = self.get_tile(row, col)
        if not tile.is_covered():
            return 0

        tile_value = tile.clear()
        if tile_value == MINE:
            self.game_status = BoardState.LOST
            return 0

        revealed = 1
        if tile_value == 0:
            revealed += self.read_tiles(row, col)
        self.win_check()
        return revealed

    def flag_tile(self, row, col):
        return self.get_tile(row, col).flag()

    def unflag_tile(self, row, col):
        return self.get_tile(row, col).unflag()
