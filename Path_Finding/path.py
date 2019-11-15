from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

import serial
import time

ser = serial.Serial()
ser.port = 'COM1'
ser.baudrate = 19200
ser.open()

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

'''
    PathA:  analyse image once
            run A* and get path
            generate commands and send via serial port
'''

class PathA:
    def __init__(self):
        pass
    # For now, no diagonal movements
    def getCommandMovementsFromPath(self, path_coor):
        coor_len = len(path_coor)
        commands = []
        i = 0
        while i+1 < coor_len:
            cur_coor = path_coor[i]
            next_coor = path_coor[i+1]

            cur_row = cur_coor[0]
            cur_col = cur_coor[1]
            next_row = next_coor[0]
            next_col = next_coor[1]
            if next_row < cur_row:
                commands.append(1)
                # move it to north
            elif next_row > cur_row:
                commands.append(4)
                # move it south
            elif next_col > cur_col:
                commands.append(2)
                # move it right
            elif cur_col > next_col:
                commands.append(3)
                # move it left
            i += 2
        
        for command in commands:
            print("\nMoving in the direction of {}\n".format(command))
            if ser.is_open:
                ser.write(command)
                time.sleep(0.3)

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
'''
    PathB:  analyse image once
            run A* and get path
            generate next command
            wait 5s and loop(analyse again to get new state)
'''
class PathB:
    def __init__(self):
        pass