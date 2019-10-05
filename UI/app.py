import sys
import serial

'''
W - North
D - East
S - South
A - West
Q - North_West
Z - South_West
C - South_East
E - North_East
'''

direction_cases = {
    'W': 0,
    'Q': 1,
    'A': 2,
    'Z': 3,
    'S': 4,
    'C': 5,
    'D': 6,
    'E': 7,
}

ser = serial.Serial()
ser.port = 'COM1'
ser.baudrate = 19200
ser.open()

print("Welcome to the Automagic Maze secret control panel. Here, you can control the platform with special commands")
print("WASD")
print("Q to move to upper left corner")
print("E to move to upper right corner")
print("Z to move to lower left corner")
print("C to move to lower right corner\n")

while True:
    user_input = str(input().upper())
    direction = direction_cases.get(user_input, 'Invalid input')
    if direction == "Invalid input":
        print("{}: Please try again with a different command \n".format(direction))
    else: 
        print("\nMoving in the direction of {}\n".format(direction))
        if ser.is_open
            ser.write(direction)

ser.close()
