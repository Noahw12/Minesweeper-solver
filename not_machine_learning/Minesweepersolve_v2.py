import pyautogui as pag
import cv2
import numpy as np
from collections import deque
from enum import Enum
from mss import mss

#Pause inbetween pyautogui operations = 0,0001 seconds
pag.PAUSE = 0.0001

class Tile:
    def __init__(self, coordinate): 
        self.value = -1
        self.coordinate = coordinate
        
    def readTile(self):
        
        with mss() as sct:
            x ,y ,width ,height = self.coordinate
            region = {
                    "top": y,
                    "left": x,
                    "width": width,
                    "height": height
                }
                # The 'grab' method captures the screen region 
            sct_img = sct.grab(region)  
            tileArr = np.array(sct_img)[:,:,:3] 
            
            self.find_average_color(sct_img)
        
            
        if self.matchTile('img/0tile.png', tileArr):
            self.value = 0
        if self.matchTile('img/Blanktile.png', tileArr):
            return
        elif self.matchTile('img/1tile.png', tileArr):
            self.value = 1
        elif self.matchTile('img/2tile.png', tileArr):
            self.value = 2
        elif self.matchTile('img/3tile.png', tileArr):
            self.value = 3
        elif self.matchTile('img/4tile.png', tileArr):
            self.value = 4
        elif self.matchTile('img/Mine.png', tileArr):
            self.value = 9
        elif self.matchTile('img/5tile.png', tileArr):
            self.value = 5
        elif self.matchTile('img/6tile.png', tileArr):
            self.value = 6
        elif self.matchTile('img/7tile.png', tileArr):
            self.value = 7
        elif self.matchTile('img/8tile.png', tileArr):
            self.value = 8
        else:
            print("Tile not recognized, Exiting...")
            exit()
    
    
    def find_average_color(self,screen_grab):
        # Load the image
        img_array = np.array(screen_grab)
        mean_bgr = np.mean(img_array[..., :3], axis=(0, 1))
        print(mean_bgr)
    
    def matchTile(self, template, tile):
        template = cv2.imread(template, cv2.IMREAD_COLOR)

        # Perform template matching
        res = cv2.matchTemplate(tile, template, cv2.TM_CCOEFF_NORMED)

        # Set a threshold for detecting a match
        threshold = 0.7
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val >= threshold:
            return True
        return False
    
    def getValue(self):
        return self.value
       
    def clear(self):
        pag.click(self.coordinate)
        self.value = self.readTile()
    
    def flag(self):
        pag.rightClick(self.coordinate)
        self.value = 10
        
class BoardState(Enum):
    ONGOING = 0
    WON = 1
    LOST = 2

class Board:
    def __init__(self):
        self.game_status = BoardState.ONGOING
        self.tiles = self.createBoard()
        self.mines = self.getMines() 
        
    def createRows(self, coordinates, n):
        tiles = [Tile(coordinate) for coordinate in coordinates]
        return np.array([tiles[i:i + n] for i in range(0, len(tiles), n)])
            
    def createBoard(self): 
        coordinates = list(pag.locateAllOnScreen('img/BlankTile.png', confidence=0.875))
        
        if len(coordinates) == 64:
            return self.createRows(coordinates, 8)
        elif len(coordinates) == 256:
            return self.createRows(coordinates, 16)
        else:
            return self.createRows(coordinates, 30)
    
    def getMines(self):
        if len(self.tiles) == 8: #Beginner
            return 10
        elif len(self.tiles) == 16: #Intermediate
            return 40
        else:          #Expert
            return 99
        
    def getTile(self, row, col):
        return self.tiles[row][col]
        
    def getTileValues(self):
        return np.array([[tile.getValue() for tile in row] for row in self.tiles])
        
    def winCheck(self): 
        if np.any(self.getTileValues() == -1):
            self.game_status = BoardState.WON
    
    def niceprint(self):
        for row in self.tiles:
            niceRow = ''
            for tile in row:
                if tile.getValue() == -1:
                    niceRow += '?' + ' '
                elif tile.getValue() == 10:
                    niceRow += 'F' + ' '
                else:
                    niceRow += str(tile.getValue()) + ' '  
            print(niceRow)
            
    def readTiles(self, row, col):
        self.getTile(row,col).readTile()
        tileVal = self.getTile(row,col).getValue()
        
        if tileVal == 0:
            rows, cols = len(self.tiles), len(self.tiles[0])
            queue = deque([(row, col)])
            visited = set([(row, col)])
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  #Up, Down, Left, Right
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]  #Top-left, Top-right, Bottom-left, Bottom-right
            
            while queue:
                current_row, current_col = queue.popleft()
            
                for dx, dy in directions:
                    nx, ny = current_row + dx, current_col + dy

                    # Check bounds and if the tile has not been visited
                    if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                        self.getTile(nx,ny).readTile()  # Process the tile
                        if self.getTile(nx,ny).getValue() == 0:
                            visited.add((nx, ny))
                            queue.append((nx, ny))
        elif tileVal == 9:
            self.game_status = BoardState.LOST
            print("\n")
            self.niceprint()
            exit()
            

    def clearTile(self, row, col):
        self.getTile(row,col).clear()
        self.readTiles(row, col)
    
    def flagTile(self, row, col):
        self.getTile(row,col).flag()
        
    def guess(self):
        hidden_array = np.array([[tile.getValue() == -1 for tile in row] for row in self.tiles])
        indices = np.argwhere(hidden_array)
        selected_index = indices[np.random.choice(len(indices))]
        self.clearTile(selected_index[0], selected_index[1])
        
    def resetGame(self):
        loss = pag.locateOnScreen('Lose.png', confidence=0.875)
        if loss != None:
            pag.click(loss)
            print("Loss")
        win = pag.locateOnScreen('Win.png', confidence=0.875)
        if win != None:
            pag.click(win)
            print("Win")
        self.tiles = self.createBoard()
        self.mines = self.getMines() 
        self.game_status = BoardState.ONGOING
    
    def start(self):
        
        self.clearTile(-1, -1) 
        if self.getTile(-1,-1).readTile() == 0: # checks if lower right tileColor is 0tile
            return
        else:
            self.clearTile(0, -1) 
            if self.getTile(0,-1).readTile() == 0: #checks upper right tileColor
                return
            else:
                self.clearTile(-1,0) 
                if self.getTile(-1,0).readTile() == 0: #checks lower left tileColor
                    return
                else:
                    self.clearTile(0,0) 
                    if self.getTile(0,0).readTile() == 0: #checks upper left tileColor
                        return
                    
                    
    def solver(self): 
        self.start()
        while True:
            while True:
                if self.winCheck():
                    print('AI Wins')
                    self.resetGame()
                else:
                    self.guess()
                    break

board = Board()

board.solver()  