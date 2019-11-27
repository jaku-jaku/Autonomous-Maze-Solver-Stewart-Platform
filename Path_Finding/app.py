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

def find_path(maze, start, end):
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
    count = 1
    cleaned_commands = [['a', 2]]
    i = 0
    while (i + 1 < len(commands)):
        if commands[i] == commands[i+1]:
            count += 0.2
        else:
            cleaned_commands.append([commands[i], count])
            count = 1
        i += 1
    cleaned_commands.append([commands[len(commands) - 1], count])
    cleaned_commands.append(['a', count])
    print(cleaned_commands)
    sendCommands(cleaned_commands, ser)
    ser.close()
