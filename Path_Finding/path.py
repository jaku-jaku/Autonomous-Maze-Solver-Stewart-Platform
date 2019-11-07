from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

def getMovementCommand():
    

def getPath(self, matrix, start, end):
    # matrix = [
    # [1, 1, 1],
    # [1, 0, 1],
    # [1, 1, 1]
    # ]
    grid = Grid(matrix=matrix)

    # start = grid.node(0, 0)
    # end = grid.node(2, 2)

    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
    path, runs = finder.find_path(start, end, grid)
    return path
    # print('operations:', runs, 'path length:', len(path), 'path: ', path)        

'''
W - North - 1
D - East - 2
A - West - 3
S - South - 4
E - North_East - 5
Q - North_West - 6
C - South_East - 7
Z - South_West - 8
'''

# For now, no diagonal movements
def getCommandMovementsFromCoordinate(curr, next):
    cur_row = curr[0]
    cur_col = curr[1]
    next_row = next[0]
    next_col = next[1]
    if next_row < cur_row:
        # move it to north
        
    else if next_row > cur_row:
        # move it south
    else if next_col > cur_col:
        # move it right
    else if cur_col > next_col:
        # mvoe it left
