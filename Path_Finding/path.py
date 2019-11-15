from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

'''
    PathA:  analyse image once
            run A* and get path
            generate commands and send via serial port
'''

class PathA:
    def __init__(self):
        pass

    # For now, no diagonal movements
    def getCommandMovementsFromPath(self, path_coor, invert):
        coor_len = len(path_coor)
        commands = []
        i = 0
        while i+1 < coor_len:
            cur_coor = path_coor[i]
            next_coor = path_coor[i+1]

            cur_row = cur_coor[1]
            cur_col = cur_coor[0]
            next_row = next_coor[1]
            next_col = next_coor[0]
            if next_row < cur_row:
                # move it to north
                if invert:
                    commands.append('4')
                else:
                    commands.append('1')
                    pass
            elif next_row > cur_row:
                # move it south
                if invert:
                    commands.append('1')
                else:
                    commands.append('4')
            elif next_col > cur_col:
                # move it right
                commands.append('2')
            elif cur_col > next_col:
                # move it left
                commands.append('3')
            i += 1
        return commands

    def getPath(self, matrix, start, end):
        grid = Grid(matrix=matrix)
        start = grid.node(start[0], start[1])
        end = grid.node(end[0], end[1])

        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        path, runs = finder.find_path(start, end, grid)

        print(grid.grid_str(path=path, start=start, end=end))
        return path


'''
    PathB:  analyse image once
            run A* and get path
            generate next command
            wait 5s and loop(analyse again to get new state)
'''
class PathB:
    def __init__(self):
        pass


