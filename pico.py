from machine import Pin, ADC
import utime

xAxis = ADC(Pin(27))              # Joystick X-axis
button = Pin(17, Pin.IN, Pin.PULL_UP)

LEFT_THRESHOLD  = 20000
RIGHT_THRESHOLD = 45000

while True:
    xValue = xAxis.read_u16()
    buttonValue = button.value()

    if buttonValue == 0:
        print(" ", end="")  # Send space
    else:
        if xValue < LEFT_THRESHOLD:
            print("a", end="")
        elif xValue > RIGHT_THRESHOLD:
            print("d", end="")
        # Otherwise do nothing (no output)

    utime.sleep(0.2)
