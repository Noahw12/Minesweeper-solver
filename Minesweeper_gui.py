import pygame
import Minesweeper as ms
import time

class GUI:
    def __init__(self, board):
        pygame.init()
        pygame.mixer.quit()
        self.running = True
        self.board = board
        self.tileSize = 32
        self.width = self.board.cols * self.tileSize
        self.height = self.board.rows * self.tileSize
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.selectedSurface = pygame.Surface((self.tileSize, self.tileSize))
        self.selectedSurface.fill((128, 0, 128, 128))
        self.images = self.init_images()
        self.clock = pygame.time.Clock().tick(30)
        
        
    def init_images(self):
        # Load images and store them as attributes
        images = {
            9: pygame.image.load('img/BlankTile.png'),   # Assuming 'blank' for covered tiles
            0: pygame.image.load('img/0Tile.png'),
            1: pygame.image.load('img/1Tile.png'),
            2: pygame.image.load('img/2Tile.png'),
            3: pygame.image.load('img/3Tile.png'),
            4: pygame.image.load('img/4Tile.png'),
            5: pygame.image.load('img/5Tile.png'),
            6: pygame.image.load('img/6Tile.png'),
            7: pygame.image.load('img/7Tile.png'),
            8: pygame.image.load('img/8Tile.png'),
            -1: pygame.image.load('img/bomb.png'),
            10: pygame.image.load('img/flag.png')
        }
        for key in images:
            images[key] = pygame.transform.scale(images[key], (32, 32))
            
        return images
    
    def display(self):
        self.screen.fill((0, 0, 0))  # Clear the screen
        
        for y, row in enumerate(self.board.getTileValues()):
            for x, tileVal in enumerate(row):
                image = self.images[tileVal]  # Directly access the image based on tileVal
                self.screen.blit(image, (x * self.tileSize, y * self.tileSize))
    
        pygame.display.flip()  # Update the full display Surface to the screen
        
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False   
                    
board = ms.Board(ms.Difficulty.EXPERT)
board.firstClear(5,5)
gui = GUI(board)
board.nicePrint()
print('\n')
board.nicePrintTrueValue()
while gui.running:
    gui.handle_events()
    gui.display()
    board.guess()
    time.sleep(0.1)
    