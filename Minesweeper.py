import numpy as np
from collections import deque
from enum import Enum

class Tile:
    def __init__(self): 
        self.value = 9
        self.trueValue = np.nan
    
    def getValue(self):
        return self.value
    
    def getTrueValue(self):
        return self.trueValue
    
    def setTrueValue(self, newValue):
        self.trueValue = newValue
       
    def clear(self):
        self.value = self.trueValue
    
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
        self.tiles = self.createBoard()
        self.game_status = BoardState.ONGOING
            
            
    def get_dimensions(self, diff):
        if diff == Difficulty.BEGINNER:
            return 8, 8, 10
        elif diff == Difficulty.INTERMEDIATE:
            return 16, 16, 40
        elif diff == Difficulty.EXPERT:
            return 16, 30, 99

    def createBoard(self): 
        return [[Tile() for _ in range(self.cols)] for _ in range(self.rows)]
            
    def firstClearSmallSafeZone(self, row, col):
        mines_placed = 0
        while mines_placed < self.mines:
            r, c = np.random.randint(0, self.rows - 1), np.random.randint(0, self.cols - 1)
            if (r, c) != (row, col):
                self.getTile(r,c).setTrueValue(-1)
                mines_placed += 1

        for r in range(row):
            for c in range(col):
                if self.getTile(r,c).getTrueValue() == -1:
                    continue
                count = sum(self.getTile(r + dr, c + dc).getTrueValue() == -1 for dr in range(-1, 2)
                            for dc in range(-1, 2) if 0 <= r + dr < self.rows and 0 <= c + dc < self.cols)
                self.getTile(r,c).setTrueValue(count)
        
        self.clearTile(row,col)
                
    #TODO antalet bomber är fel
    def firstClear(self, row, col):
        #Place mines ensuring first click is safe
        safe_zone = {(row + i, col + j) for i in range(-1, 2) 
                     for j in range(-1, 2) if 0 <= row + i < self.rows and 0 <= col + j < self.cols}
        
        mines_placed = 0
        while mines_placed < self.mines:
            r, c = np.random.randint(0, self.rows - 1), np.random.randint(0, self.cols - 1)
            if (r, c) not in safe_zone:
                self.tiles[r][c].setTrueValue(-1)
                mines_placed += 1

        # Calculate adjacent mines for each cell
        for r in range(self.rows):
            for c in range(self.cols):
                if self.tiles[r][c].getTrueValue() == -1:
                    continue
                count = sum(self.getTile(r + dr, c + dc).getTrueValue() == -1 for dr in range(-1, 2) 
                            for dc in range(-1, 2) if 0 <= r + dr < self.rows and 0 <= c + dc < self.cols)
                self.getTile(r,c).setTrueValue(count)
        
        self.clearTile(row,col)
        
    def getTile(self, row, col):
        return self.tiles[row][col]
        
    def getTileValues(self):
        return np.array([[tile.getValue() for tile in row] for row in self.tiles])
        
    def winCheck(self): 
        if not np.any(self.getTileValues() == 9):
            self.game_status = BoardState.WON
    
    def nicePrint(self):
        for row in self.tiles:
            niceRow = ''
            for tile in row:
                if tile.getValue() == 9:
                    niceRow += '?' + ' '
                elif tile.getValue() == 10:
                    niceRow += 'F' + ' '
                elif tile.getValue() == -1:
                    niceRow += 'B' + ' '
                else:
                    niceRow += str(tile.getValue()) + ' '  
            print(niceRow)
            
    def nicePrintTrueValue(self):
        for row in self.tiles:
            niceRow = ''
            for tile in row:
                if tile.getTrueValue() == 10:
                    niceRow += 'F' + ' '
                elif tile.getTrueValue() == -1:
                    niceRow += 'B' + ' '
                else:
                    niceRow += str(tile.getTrueValue()) + ' '  
            print(niceRow)
            
    def readTiles(self, row, col):
        tileVal = self.getTile(row,col).getValue()
        
        if tileVal == 0:
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
                        self.clearTile(nx,ny)  #Process the tile
                        if self.getTile(nx,ny).getValue() == 0:
                            visited.add((nx, ny))
                            queue.append((nx, ny))
        elif tileVal == -1:
            self.game_status = BoardState.LOST
            
    def clearTile(self, row, col):
        if self.getTile(row,col).getValue() == 9:
            self.getTile(row,col).clear()
            self.readTiles(row, col)
    
    def flagTile(self, row, col):
        self.getTile(row,col).flag()
        
    def guess(self):
        hidden_array = np.array([[tile.getValue() == 9 for tile in row] for row in self.tiles])
        indices = np.argwhere(hidden_array)
        selected_index = indices[np.random.choice(len(indices))]
        self.clearTile(selected_index[0], selected_index[1])
        