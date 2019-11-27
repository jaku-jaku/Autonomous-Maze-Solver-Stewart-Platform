import serial
import time

from path import PathA
from Astar import Astar
from config import MAX_HEAT_MAP_VALUE, MAX_HEAT_MAP_WEIGHT, COUNT_DELAY, COUNT_DELAY_REPEAT

def sendCommands(commands, ser):
    for command in commands:
        print("\nMoving in the direction of {}\n".format(command[0]))
        if ser.is_open:
            print("\nSerial port is open and sending the following command")
            ser.write(command[0].encode())
            time.sleep(command[1])

def find_path(maze, start, end, heat_map = None):
    if heat_map is not None:
        # run custom code here, for dual code support
        ASTAR = Astar(MAX_HEAT_MAP_VALUE=MAX_HEAT_MAP_VALUE, HEAT_MAP_WEIGHT=MAX_HEAT_MAP_WEIGHT)
        map_grid, sNode, eNode  = ASTAR.genMap(maze, start, end, PATH_VALUE=1, heat_map=heat_map)
        sNode.printSelf()
        eNode.printSelf()
        print('--- RESULT ---')
        path = ASTAR.findPath(map_grid, sNode, eNode, ALLOW_DIAG=False)
        pathArray = ASTAR.extractPath(path)
        ASTAR.printMaze(map_grid, pathArray)
        return pathArray
    else:
        print('Ima end these coordinates')
        print(start, end)
        pathA = PathA()
        return pathA.getPath(maze, start, end)

def send_path(path, tilt_angle):
    print('Below is the tilt angle')
    print(tilt_angle)
    pathA = PathA()
    commands = pathA.getCommandMovementsFromPath(path, tilt_angle)
    ser = serial.Serial()
    ser.port = '/dev/tty.usbmodem141403'
    ser.baudrate = 9600
    ser.open()
    count = COUNT_DELAY
    cleaned_commands = [['a', 2]]
    i = 0
    while (i + 1 < len(commands)):
        if commands[i] == commands[i+1]:
            count += COUNT_DELAY_REPEAT
        else:
            cleaned_commands.append([commands[i], count])
            count = COUNT_DELAY
        i += 1
    cleaned_commands.append([commands[len(commands) - 1], count])
    cleaned_commands.append(['a', count])
    print(cleaned_commands)
    sendCommands(cleaned_commands, ser)
    ser.close()
