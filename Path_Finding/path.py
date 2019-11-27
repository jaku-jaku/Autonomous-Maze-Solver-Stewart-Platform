from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

'''
    PathA:  analyse image once
            run A* and get path
            generate commands and send via serial port
    'Neutral': 'a',
    '0': 'N' : 'b'
    '90': 'E': 'c',
    '270': 'W': 'd'
    '180': 'S': 'e',
    '45': 'NE': 'f'
    '315': 'NW': 'g',
    '135': 'SE': 'h'
    '225': 'SW': 'i'
    '22.5': 'NNE': 'j'
    '67.5': 'ENE': 'k',
    '337.5': 'NNW': 'l':
    '292.5': 'WNW': 'm',
    '157.5': 'SSE': 'n',
    '112.5': 'ESE': 'o',
    '202.5': 'SSW': 'p',
    '247.5': 'WSW': 'q',
'''

class PathA:
    def __init__(self):
        pass

    def getNearestCardinal(self, angle, offset):
        deviation = 12.25
        true_angle = (angle + offset) % 360
        cardinal_commands = {
            '0': 'b',
            '22.5': 'j',
            '45': 'f',
            '67.5': 'k',
            '90': 'c',
            '112.5': 'o',
            '135': 'h',
            '157.5': 'n',
            '180': 'e',
            '202.5': 'p',
            '225': 'i',
            '247.5': 'q',
            '270': 'd',
            '292.5': 'm',
            '315': 'g',
            '337.5': 'l',
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


