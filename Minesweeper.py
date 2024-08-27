import numpy as np
from collections import deque
from enum import Enum

class Tile:
    def __init__(self): 
        self.value = 9
        self.trueValue = np.nan
    
    def get_value(self):
        return self.value
    
    def get_true_value(self):
        return self.trueValue
    
    def set_true_value(self, newValue):
        self.trueValue = newValue
       
    def clear(self):
        if self.trueValue == -1:
            return self.trueValue
        else:
            self.value = self.trueValue
            return self.value
    
    def flag(self):
        self.value = 10
        
class Difficulty(Enum):
    BEGINNER = 1
    INTERMEDIATE = 2
    EXPERT = 3
    
class BoardState(Enum):
    ONGOING = 0
    WON = 1
    LOST = 2

class Board:
    def __init__(self, diff):
        self.diff = diff
        self.rows, self.cols, self.mines = self.get_dimensions(diff)
        self.tiles = self.create_board()
        self.game_status = BoardState.ONGOING
        #Make the first move
        self.first_clear(np.random.randint(0,self.cols), 
                              np.random.randint(0,self.rows))
            
            
    def get_dimensions(self, diff):
        if diff == Difficulty.BEGINNER:
            return 8, 8, 10
        elif diff == Difficulty.INTERMEDIATE:
            return 16, 16, 40
        elif diff == Difficulty.EXPERT:
            return 16, 30, 99

    def create_board(self): 
        return [[Tile() for _ in range(self.cols)] for _ in range(self.rows)]
            
    def first_clear_small_safe_zone(self, row, col):         
        all_positions = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        available_positions = [pos for pos in all_positions if pos != (row,col)]
        np.random.shuffle(available_positions)
        for r, c in available_positions[:self.mines]:
            self.get_tile(r,c).set_true_value(-1)

        for r in range(row):
            for c in range(col):
                if self.get_tile(r,c).get_true_value() == -1:
                    continue
                count = sum(self.get_tile(r + dr, c + dc).get_true_value() == -1 for dr in range(-1, 2)
                            for dc in range(-1, 2) if 0 <= r + dr < self.rows and 0 <= c + dc < self.cols)
                self.get_tile(r,c).set_true_value(count)
        
        self.clear_tile(row,col)
                
    def first_clear(self, row, col):
        #Place mines ensuring first click is safe
        safe_zone = {(row + i, col + j) for i in range(-1, 2) 
                     for j in range(-1, 2) if 0 <= row + i < self.rows and 0 <= col + j < self.cols}
        
        all_positions = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        available_positions = [pos for pos in all_positions if pos not in safe_zone]
        np.random.shuffle(available_positions)
        for r, c in available_positions[:self.mines]:
            self.get_tile(r,c).set_true_value(-1)

        #Calculate adjacent mines for each cell
        for r in range(self.rows):
            for c in range(self.cols):
                if self.tiles[r][c].get_true_value() == -1:
                    continue
                count = sum(self.get_tile(r + dr, c + dc).get_true_value() == -1 for dr in range(-1, 2) 
                            for dc in range(-1, 2) if 0 <= r + dr < self.rows and 0 <= c + dc < self.cols)
                self.get_tile(r,c).set_true_value(count)
        
        self.clear_tile(row,col)
        
    def get_tile(self, row, col):
        return self.tiles[row][col]
        
    def get_tile_values(self):
        return np.array([[tile.get_value() for tile in row] for row in self.tiles])
    
    def get_true_tile_values(self):
        return np.array([[tile.get_true_value() for tile in row] for row in self.tiles])
        
    def win_check(self): 
        if np.count_nonzero(self.get_tile_values() == 9) == self.mines:
            self.game_status = BoardState.WON
    
    def nice_print(self):
        for row in self.tiles:
            niceRow = ''
            for tile in row:
                if tile.get_value() == 9:
                    niceRow += '?' + ' '
                elif tile.get_value() == 10:
                    niceRow += 'F' + ' '
                elif tile.get_value() == -1:
                    niceRow += 'B' + ' '
                else:
                    niceRow += str(tile.get_value()) + ' '  
            print(niceRow)
            
    def nice_print_true_values(self):
        for row in self.tiles:
            niceRow = ''
            for tile in row:
                if tile.get_true_value() == 10:
                    niceRow += 'F' + ' '
                elif tile.get_true_value() == -1:
                    niceRow += 'B' + ' '
                else:
                    niceRow += str(tile.get_true_value()) + ' '  
            print(niceRow)
    
    def read_tiles(self, row, col):
        
        queue = deque([(row, col)])
        visited = set([(row, col)])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  #Up, Down, Left, Right
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]  #Top-left, Top-right, Bottom-left, Bottom-right
            
        while queue:
            current_row, current_col = queue.popleft()
            
            for dx, dy in directions:
                nx, ny = current_row + dx, current_col + dy

                #Check bounds and if the tile has not been visited
                if 0 <= nx < self.rows and 0 <= ny < self.cols and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    if self.get_tile(nx,ny).clear() == 0:  #Process the tile
                        queue.append((nx, ny))
            
            
    def clear_tile(self, row, col):
        tile_value = self.get_tile(row,col).clear()
        if tile_value == -1:
            self.game_status = BoardState.LOST
        elif tile_value == 0:
            self.read_tiles(row, col)
        self.win_check()
            
    def flag_tile(self, row, col):
        self.get_tile(row,col).flag()
        
    def guess(self):
        hidden_array = np.array([[tile.get_value() == 9 for tile in row] for row in self.tiles])
        indices = np.argwhere(hidden_array)
        selected_index = indices[np.random.choice(len(indices))]
        self.clear_tile(selected_index[0], selected_index[1])
        