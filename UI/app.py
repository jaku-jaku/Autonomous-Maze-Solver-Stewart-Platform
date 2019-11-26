import sys
import serial

def getch():
    import termios
    import sys, tty
    def _getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    return _getch()

'''
W - North - 1
D - East - 2
A - West - 3
S - South - 4
E - North_East - 5
Q - North_West - 6
C - South_East - 7
Z - South_West - 8

COMMANDS TO REPRODUCE BUG
WEST, SOUTH, WEST, WEST, NORTH WEST, NORTH WEST, NORTH, EAST, SOUTH, MOTOR
6, 11, -5, 11, 28, 28, 16, -11, -16,
'''

direction_cases = {
    'P': '0',
    'W': '1',
    'D': '2',
    'A': '3',
    'S': '4',
    'E': '5',
    'Q': '6',
    'C': '7',
    'Z': '8',
}

ser = serial.Serial('/dev/tty.usbmodem143303', 9600)

print("Welcome to the Automagic Maze secret control panel. Here, you can control the platform with special commands")
print("WASD")
print("Q to move to upper left corner")
print("E to move to upper right corner")
print("Z to move to lower left corner")
print("C to move to lower right corner\n")

while True:
    user_input = str(getch().upper())
    direction = direction_cases.get(user_input, 'Invalid input')
    if direction == "Invalid input":
        print("{}: Please try again with a different command \n".format(direction))
    else:
        print("\nMoving in the direction of {}\n".format(direction))
        if ser.is_open:
            ser.write(direction.encode())
