import serial
import sys
s = serial.Serial("/dev/cu.usbmodem1413", sys.argv[1])
while True:
    print s.readline(), 
