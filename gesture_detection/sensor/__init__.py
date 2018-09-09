
import serial
import numpy as np


class Sensor:
    def __init__(self, port = '/dev/tty0', freq = 9600, timeout = 3):
        self.port = port
        self.freq = freq
        self.timeout = timeout
        print('------------------------------ device initializing ------------------------------')
        self.device = serial.Serial( port, freq, timeout = timeout)
        print('device {} is on {} with frequency {} Hz.'.format(self.device.name, self.device.port, self.device.baudrate))
        print('------------------------------ device initialized -------------------------------')
        self.flush()
    def read(self):
        # return ( type : np.array, shape: (9,1))
        # meaning accelX accelY accelZ gyroX gyroY gyroZ magneticX magneticY magneticZ define in opt
        data = self._read()
        data = np.array(data)
        return data

    def _read(self):
        string = self.device.readline()
        string = string.strip()
        data = []
        print(string)
        for part in string.split('|'):
            print(part)
            for value in part.split(' ')[1:]:
                data.append(float(value))
        return data

    def flush(self):
        self.device.flushInput()
        

if __name__ == '__main__':
    sensor = Sensor()
