import pyautogui as pag
import time
import itertools as it
import numpy as np
from collections import deque
from copy import deepcopy
from sys import exit
from mss import mss


pag.PAUSE = 0.001 # pause inbetween pyautogui operations = 0,001 seconds

# 6-tile
#[159.68333333 167.03095238 173.94047619]
#166.88492063492063 avg
# 145 < tileColor[0] < tileColor[1] < tileColor[2] < 185: # 6-tile, problem?

def readTile(coord):
   tileArr = np.array(pag.screenshot(region = coord))
#    with mss() as sct:
#         x ,y ,width ,height = coord
#         region = {
#             "top": y,
#             "left": x,
#             "width": width,
#             "height": height
#             }
#         # The 'grab' method captures the screen region defined by the dictionary
#         sct_img = sct.grab(region)
                
#         # Convert the captured image to a NumPy array
#         # Note: MSS returns BGRA by default (Blue, Green, Red, Alpha)
#         tileArr = np.array(sct_img)
   tileColor = np.mean(tileArr, (0,1))
   if 185 < np.average(tileColor) < 195: # checks if '?' shade of gray
        return '?'
   elif tileColor[0] == tileColor[1] == tileColor[2]: # checks if gray
        return 0
   elif tileColor[0] == tileColor[1] and 185 < tileColor[2]: #checks for 1-tile
        return 1
   elif tileColor[0] == tileColor[2] and 160 < tileColor[1]: #checks for 2-tile
        return 2
   elif tileColor[1] == tileColor[2] and 190 < tileColor[0]: #checks for 3-tile
        return 3
   elif 145 < np.average(tileColor) < 155: # checks for 4-tile
        return 4
   elif  isMine(int(coord[0]),int(coord[1])): #checks for mine
        return 'M'
   elif 135 < np.average(tileColor) < 145: # checks for 5-tile
        return 5  
   elif 160 < np.average(tileColor) < 180: # checks for 6-tile, Problem
        print('6-tile')
        return 6
   else:  # its a 7-tile
        print('7-tile')
        return 7

def createRows(lst, n):
    max(1, n)
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def createBoard(boardcoords):
    if len(boardcoords) == 64:
        return np.array(createRows(boardcoords, 8), dtype=object)
    elif len(boardcoords) == 256:
        return np.array(createRows(boardcoords, 16), dtype=object)
    else:
        return np.array(createRows(boardcoords, 30), dtype=object)
        

def niceprint(board):
    for row in board:
        niceRow = ''
        for i in row:
            niceRow += str(i) + ' '   
        print(niceRow)

def probabilitylist(coordLst):
    problist = []
    for _ in coordLst:
        problist.append(0)
    return createBoard(problist)

def visitList(coordLst):
    visLst = []
    for _ in coordLst:
        visLst.append(False)
    return createBoard(visLst)

def newBoard(coordLst):
    tileLst = []
    for _ in coordLst:
        tileLst.append('?')
    return createBoard(tileLst)

def getMines(coordlst):
    if len(coordlst) == 64:    #beginner
        return 10
    elif len(coordlst) == 256: #intermediate
        return 40
    else:                   #expert
        return 99

# global variables that are needed, enkapsulera med OOP
coordLst = list(pag.locateAllOnScreen('BlankTile.png', confidence=0.875))
board = newBoard(coordLst)
boardCoords = createBoard(coordLst)
visLst = visitList(coordLst)
mines = getMines(coordLst)
lastRowIndex = len(board) - 1
lastTileIndex = (len(board[0]) - 1)

def isValidBFS(row, tile):    
    if row < 0 or tile < 0 or row > lastRowIndex or tile > lastTileIndex:
        return False
    elif visLst[row][tile]:
        return False
    else:
        return True

def isValid(row, tile):
    if row < 0 or tile < 0 or row > lastRowIndex or tile > lastTileIndex:
        return False
    else:
        return True

# effektivisera, väldigt långsam, bottleneck för hela programmet
# kan vara för att den bara stannar vid '?', lägg till condition för inte 0
def readtiles(row, tile):
    tileVal = readTile(boardCoords[row][tile])
    queue = deque()
    vectorRow = [-1, 0, 1, 0]
    vectorTile = [0, 1, 0, -1]
    if tileVal == 0:
        queue.append((row,tile))
        visLst[row][tile] = True
        while len(queue) > 0:
            coord = queue.popleft()
            y = coord[0]
            x = coord[1]
            board[y][x] = readTile(boardCoords[y][x])
            for i in range(4):
                adjRow = y + vectorRow[i]
                adjTile = x + vectorTile[i]
                if isValidBFS(adjRow, adjTile) and readTile(boardCoords[adjRow][adjTile]) != '?': 
                   visLst[adjRow][adjTile] = True
                   queue.append((adjRow, adjTile))        
    elif tileVal == 'M':
        print('BOOM')
        exit()      #tillfällig lösning   
    else:
        visLst[row][tile] = True
        board[row][tile] = tileVal


def adjOp(row, tile, op):
    count = 0
    flags = 0
    coordlist = []
    adjLst = [(-1,-1,),(-1,0,),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] 
    for i in adjLst:
        adjRow = row + i[0]
        adjTile = tile + i[1]
        if isValid(adjRow, adjTile) and board[adjRow][adjTile] in ['F', '?']:
            if board[adjRow][adjTile] == 'F':
                flags += 1
            elif board[adjRow][adjTile] == '?':
                count += 1 
                coordlist.append((adjRow,adjTile))
    if op == 'flag' and coordlist != [] and count == board[row][tile] - flags:
        for coord in coordlist:
            flag(coord[0],coord[1])
    if op == 'clear' and coordlist != [] and flags == board[row][tile]:
        for coord in coordlist:
            clear(coord[0],coord[1])
            

def logicTactic():
    coords = np.argwhere(((board != '?') & (board != 'F')) & (board != 0))
    for coord in coords:
       adjOp(coord[0], coord[1], 'flag')
    for coord in coords:
        adjOp(coord[0], coord[1], 'clear')


# förbättra guess funktion        
def guess():
    print('guess')
    if board[lastRowIndex][0] == '?':
        clear(lastRowIndex,0)
    elif board[0][lastTileIndex] == '?':
        clear(0, lastTileIndex)
    elif board[0][0] == '?':
        clear(0,0)
    else:                            #gör om till local probability? Ja
        #niceprint(localprob())
        coord = np.random.Generator.choice(np.argwhere(board == '?'), size=1, axis=0)
        clear(coord[0],coord[1])  
        # rng = np.random.default_rng()
        # indices = np.argwhere(board == '?')
        # coord = rng.choice(indices, size=1, axis=0)[0]
        # clear(coord[0], coord[1]) 


def localprob():
    problst = borderprob(findBorderTiles())
    coords = np.argwhere(problst == 0)
    for coord in coords:
        if visLst[coord[0]][coord[1]] == True:
            problst[coord[0]][coord[1]] = 1
        else:
            problst[coord[0]][coord[1]] = mines / len(coords) 
    return problst

def borderprob(borderTiles):
    problist = probabilitylist(coordLst) 
    adjLst = [(-1,-1,),(-1,0,),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] 
    for coord in borderTiles:
        count = 0
        flags = 0
        adjCovered = []
        for i in adjLst:
            adjRow = coord[0] + i[0]
            adjTile = coord[1] + i[1]
            if isValid(adjRow,adjTile) and board[adjRow][adjTile] == '?':
                count += 1
                adjCovered.append((adjRow,adjTile))
            elif isValid(adjRow,adjTile) and board[adjRow][adjTile] == 'F':
                flags += 1
        for i in adjCovered:
            print(flags) # flags är alltid 0? måste göra något med antalet '?' buggar ur när flagga är bordertile
            if type(board[coord[0]][coord[1]]) == int: # bug, troligen med divisionen
                problist[i[0]][i[1]] = max(problist[i[0]][i[1]], board[coord[0]][coord[1]] / max(1,(count - flags)))
    return problist


def guess1():
    print('guess')
    if board[lastRowIndex][0] == '?':
        clear(lastRowIndex,0)
    elif board[0][lastTileIndex] == '?':
        clear(0, lastTileIndex)
    elif board[0][0] == '?':
        clear(0,0)
    else:                             #gör om till local probability? Ja
        coord = np.random.choice(np.argwhere(board == '?').flatten())
        clear(coord[0],coord[1])   


def flag(row, tile):
    pag.rightClick(tuple(boardCoords[row][tile]))
    board[row][tile] = 'F'
    visLst[row][tile] = True 
    global mines # fixa
    mines -= 1
    
    
def clear(row, tile):
    pag.click(tuple(boardCoords[row][tile]))
    readtiles(row, tile) 

def start():
    clear(lastRowIndex, lastTileIndex) #clears lower right tileColor
    if readTile(boardCoords[-1][-1]) == 0: # checks if lower right tileColor is 0tile
        return
    else: 
        clear(0, lastTileIndex)
        if readTile(boardCoords[0][-1]) == 0: #checks upper right tileColor
            return
        else:
            clear(lastRowIndex,0)
            if readTile(boardCoords[-1][0]) == 0: #checks lowerleft tileColor
                return
            else:
                clear(0,0)
                if readTile(boardCoords[0][0]) == 0: #checks upperleft tileColor
                    return

def winCheck(): 
    if '?' in board:
        return False
    else:
        return True


def CSP():
    op = False
    problist = mineprobability(BoardPermutations())
    for coord in np.argwhere(problist == 0.0):
        if visLst[coord[0]][coord[1]] == False: 
            op = True
            clear(coord[0],coord[1])
    for coord in np.argwhere(problist == 1.0):
        op = True
        flag(coord[0],coord[1])
    if op == False and len(problist) > 0:
        coord = getLowProbCoord(problist)
        clear(coord[0],coord[1])


def findBorderTiles():
    coordlist = []
    for coord in np.argwhere(board == '?'):
        adjLst = [(-1,-1,),(-1,0,),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        for i in adjLst:
            adjRow = coord[0] + i[0]
            adjTile = coord[1] + i[1]
            if (adjRow,adjTile) not in coordlist:
                if isValid(adjRow,adjTile) and board[adjRow][adjTile] != '?': 
                    coordlist.append((adjRow,adjTile))
    return coordlist
    

def constraints(perm, bordertiles): 
    adjLst = [(-1,-1,),(-1,0,),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    perm2D = createBoard(perm)
    if np.count_nonzero(perm == '1') != mines: 
        return False
    for coord in bordertiles: 
        count = 0
        for i in adjLst:
            adjRow = coord[0] + i[0]
            adjTile = coord[1] + i[1]
            if isValid(adjRow,adjTile) and perm2D[adjRow][adjTile] in ['1', 'F']: 
                count += 1
        if count != perm2D[coord[0]][coord[1]] and perm2D[coord[0]][coord[1]] != 'F' :
            return False
    else:
        return True


def BoardPermutations(): 
    permlist = np.array(list(it.chain(*deepcopy(board))),dtype=object)
    indexlst = np.argwhere(permlist == '?')
    if len(indexlst) > 20:
        return []
    else:
        print(f'{len(indexlst)} CSP tiles')
        domain = ['0','1']
        boardpermutations = []
        varList = [domain for _ in indexlst]
        bordertiles = findBorderTiles()
        permutations = list(it.product(*varList))
        for i in permutations:                      # kan man generara permutationer tillsammans med constraints?
            for count, index in enumerate(indexlst): # list comprehension?
                permlist[index] = i[count]           
            if constraints(permlist, bordertiles):
                boardpermutations.append(createBoard(permlist))
    return boardpermutations


def mineprobability(boardperms):
    probLst = probabilitylist(coordLst)
    if boardperms == []:
        return []
    else:
        for perm in boardperms:
            for i in np.argwhere(perm == '1'): 
                probLst[i[0]][i[1]] = probLst[i[0]][i[1]] + 1               
            probLst = np.array(list(map(lambda x:  x/len(boardperms), probLst)), dtype=object)
    return probLst


def getLowProbCoord(problist):  
    arr = np.where(problist == 0.0, 1, problist)
    return divmod(arr.argmin(), arr.shape[1])

def isMine(x,y):
    r,g,b = pag.pixel(x+5,y+5)
    if g == b == 0 and 235 < r:
        return True
    else:
        return False

def getcolor():
    while True:
        x,y = pag.position()
        r,g,b = pag.pixel(x,y)
        print((r,g,b))
        time.sleep(0.5)

def getAverageColorTile():
    while True:
     x, y = pag.position()
     coord = (x,y,21,20)
     tilearr = np.array(pag.screenshot(region= coord))
     print(np.mean(tilearr, (0,1)))
     print(np.average(np.mean(tilearr, (0,1))))
     time.sleep(1)


def solver(): 
    start()
    while True:
        while True:
            if winCheck():
                print('AI Wins')
                exit() # temporärt
            else:
                temp = deepcopy(board) 
                logicTactic() 
                if np.all(board == temp):
                    CSP()
                    if np.all(board == temp):
                        guess()
                        break
                    else:
                        break

solver()  