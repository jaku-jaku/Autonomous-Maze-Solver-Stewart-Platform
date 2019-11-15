import serial
import time

from path import PathA

def sendCommands(commands, ser):
    for command in commands:
        print("\nMoving in the direction of {}\n".format(command[0]))
        if ser.is_open:
            print("\nSerial port is open and sending the following command")
            ser.write(command[0].encode())
            time.sleep(command[1])

def main():
    ser = serial.Serial()
    ser.port = '/dev/tty.usbmodem141303'
    ser.baudrate = 9600
    ser.open()

    matrix = [
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    # coordinates as [x, y]
    start = [1, 7]
    end = [9, 3]

    pathA = PathA()
    # print(pathA.getCommandMovementsFromPath(pathA.getPath(matrix, start, end), True))

    commands = pathA.getCommandMovementsFromPath(pathA.getPath(matrix, start, end), True)

    count = 0.3
    # [4,2,2,1,1]
    cleaned_commands = [['0', 2.5]]
    i = 0
    while (i + 1 < len(commands)):
        if commands[i] == commands[i+1]:
            count += 0.2
        else: 
            cleaned_commands.append([commands[i], count])
            count = 0.8
        i += 1
    cleaned_commands.append([commands[len(commands) - 1], count])
    cleaned_commands.append(['0', count])
    print(commands)
    print(cleaned_commands)
    sendCommands(cleaned_commands, ser)
    ser.close()

main()
'''
    W - North - 1
    D - East - 2
    A - West - 3
    S - South - 4
    E - North_East - 5
    Q - North_West - 6
    C - South_East - 7
    Z - South_West - 8

    path: [(1, 7), (1, 6), (2, 6), (3, 6), (3, 7), (3, 8), (4, 8), (5, 8), (6, 8), 
    (7, 8), (8, 8), (9, 8), (9, 7), (9, 6), (9, 5), (9, 4), (9, 3)]
    +-----------+
    |########  #|
    |#   # #   #|
    |##    # ###|
    |##    # #e#|
    |##### # #x#|
    |#   #   #x#|
    |#xxx#   #x#|
    |#s#x#  ##x#|
    |###xxxxxxx#|
    |#         #|
    |###########|
    +-----------+
    
    Not-inverted
    [1, 2, 2, 4, 4, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]
    [N, E, E, S, S, E, E, E, E, E, E, N, N, N, N, N]

    Inverted - correct orientation
    [4, 3, 3, 1, 1, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
    [S, W, W, N, N, W, W, W, W, W, W, S, S, S, S, S]

    North-South inverted only
    [4, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4]
    [S, E, E, N, N, E, E, E, E, E, E, S, S, S, S, S]
'''