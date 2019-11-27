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

direction_cases = {
    'P': 'a',
    'W': 'b',
    'D': 'c',
    'A': 'd',
    'S': 'e',
}

ser = serial.Serial('/dev/tty.usbmodem141403', 9600)

print("Welcome to the Automagic Maze secret control panel. Here, you can control the platform with special commands")
print("WASD")
print("Q to move to upper left corner")
print("E to move to upper right corner")
print("Z to move to lower left corner")
print("C to move to lower right corner\n")

while True:
    user_input = str(getch().lower())
    print("\nMoving in the direction of {}\n".format(user_input))
    if ser.is_open:
        ser.write(user_input.encode())
