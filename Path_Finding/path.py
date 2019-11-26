from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

'''
    PathA:  analyse image once
            run A* and get path
            generate commands and send via serial port
    
    '0': 'N' : '1'
    '22.5': 'NNE': 'a'
    '45': 'NE': '5'
    '67.5': 'ENE': 'b',
    '90': 'E': '2',
    '112.5': 'ESE': 'c',
    '135': 'SE': '6'
    '157.5': 'SSE': 'd',
    '180': 'S': '3',
    '202.5': 'SSW': 'e',
    '225': 'SW': '7'
    '247.5': 'WSW': 'f',
    '270': 'W': '4'
    '292.5': 'WNW': 'g',
    '315': 'NW': '8',
    '337.5': 'NNW': 'h':

'''

class PathA:
    def __init__(self):
        pass

    def getNearestCardinal(self, angle, offset):
        deviation = 12.25
        true_angle = (angle + offset) % 360
        cardinal_commands = {
            '0': '1',
            '22.5': 'a',
            '45': '5',
            '67.5': 'b',
            '90': '2',
            '112.5': 'c',
            '135': '6',
            '157.5': 'd',
            '180': '3',
            '202.5': 'e',
            '225': '7',
            '247.5': 'f',
            '270': '4',
            '292.5': 'g',
            '315': '8',
            '337.5': 'h',
        }

        for cardinal_angle, direction in cardinal_commands.items():
            cardinal_angle = float(cardinal_angle)
            lower_bound = cardinal_angle - deviation
            upper_bound = cardinal_angle + deviation
            if lower_bound < 0 :
                lower_bound = 360 + lower_bound
            if upper_bound >= 360 :
                upper_bound = upper_bound - 360
            if lower_bound <= true_angle <= upper_bound :
                return direction
        return 0

    def getCommandMovementsFromPath(self, path_coor, offset):
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
            angle = 0
            if next_row < cur_row:
                # move it to north
                angle = 0
            elif next_row > cur_row:
                # move it south
                angle = 180
            elif next_col > cur_col:
                # move it right
                angle = 90
            elif cur_col > next_col:
                # move it left
                angle = 270
            cardinal_direction = self.getNearestCardinal(angle, offset)
            commands.append(cardinal_direction)
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


