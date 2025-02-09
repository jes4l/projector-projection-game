import serial
import pyautogui

# Replace "COM6" with the actual port your Pico uses
ser = serial.Serial("COM6", 115200, timeout=1)

while True:
    # Read 1 byte at a time
    data = ser.read(1)  # returns a bytes object of length 1 (or 0 if timed out)

    if not data:
        # No data received within timeout
        continue

    char = data.decode('utf-8', errors='ignore')

    # Map the received characters to key presses
    if char == 'a':
        pyautogui.press('a')
        pyautogui.keyDown('a')
        pyautogui.keyUp('a')

    elif char == 'd':
        pyautogui.press('d')
        pyautogui.keyDown('d')
        pyautogui.keyUp('d')
    elif char == ' ':
        pyautogui.press('space')
    else:
        # Ignore other characters
        pass
