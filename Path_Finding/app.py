import serial
import time
from debug_framework import setFPS_Timer, getFPS_Timer_Elapsed_Tau
from path import PathA
from Astar import Astar
from config import MAX_HEAT_MAP_VALUE, MAX_HEAT_MAP_WEIGHT, COUNT_DELAY, COUNT_DELAY_REPEAT, END_OF_CMD_DELAY


def find_path(maze, start, end, heat_map = None):
    if heat_map is not None:
        # run custom code here, for dual code support
        ASTAR = Astar(MAX_HEAT_MAP_VALUE=MAX_HEAT_MAP_VALUE, HEAT_MAP_WEIGHT=MAX_HEAT_MAP_WEIGHT)
        map_grid, sNode, eNode  = ASTAR.genMap(maze, start, end, PATH_VALUE=1, heat_map=heat_map)
        sNode.printSelf()
        eNode.printSelf()
        print('--- RESULT ---')
        path = ASTAR.findPath(map_grid, sNode, eNode, ALLOW_DIAG=False)
        pathArray = []
        if path is not None:
            if len(path) > 1:
                pathArray = ASTAR.extractPath(path)
                ASTAR.printMaze(map_grid, pathArray)
            else:
                print("[ERROR] --X Custom Astar Cannot Compute Path")
        else:
            print("[ERROR] --X Custom Astar Cannot Compute Path")
        return pathArray
    else:
        print('Ima end these coordinates')
        print(start, end)
        pathA = PathA()
        return pathA.getPath(maze, start, end)

def send_path(path, tilt_angle, port='/dev/tty.usbmodem142303'):
    print('Below is the tilt angle')
    print(tilt_angle)
    pathA = PathA()
    commands = pathA.getCommandMovementsFromPath(path, tilt_angle)
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = 9600
    ser.open()
    count = COUNT_DELAY
    cleaned_commands = [['a', 2]]
    i = 0
    while (i + 1 < len(commands)):
        if commands[i] == commands[i+1]:
            count +=  COUNT_DELAY_REPEAT
        else:
            cleaned_commands.append([commands[i], count])
            count = COUNT_DELAY
        i += 1
    cleaned_commands.append([commands[len(commands) - 1], count+END_OF_CMD_DELAY])
    cleaned_commands.append(['a', END_OF_CMD_DELAY])
    print(cleaned_commands)
    sendCommands(cleaned_commands, ser)
    ser.close()

def sendCommands(commands, ser):		
    for command in commands:		
        print("\nMoving in the direction of {}\n".format(command[0]))		
        if ser.is_open:		
            print("\nSerial port is open and sending the following command")		
            ser.write(command[0].encode())		
            time.sleep(command[1])		

# def sendCommands(commands, ser):
#     setFPS_Timer('command',0,timePeriod=0)
#     for command in commands:
#         print("\nMoving in the direction of {}\n".format(command[0]))
#         if ser.is_open:
#             print("\nSerial port is open and sending the following command")
#             ser.write(command[0].encode())
#             setFPS_Timer('command',0,timePeriod=command[1])
#             while True:
#                 if getFPS_Timer_Elapsed_Tau('command'):
#                     break
#             # time.sleep(command[1])
