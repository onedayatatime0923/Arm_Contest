
import numpy as np
import serial


class Sensor:
    def __init__(self, device = '/dev/tty0', freq = 9600, timeout = 3):
        print('------------------------------ device initializing ------------------------------')
        self.device = serial.Serial( device, freq, timeout = timeout)
        print('device {} is on {} with frequency {} Hz.'.format(self.device.name, self.device.port, self.device.baudrate))
        print('------------------------------ device initialized -------------------------------')
        self.data = {'A':[0 for i in range(3)],
                     'G':[0 for i in range(3)],
                     'M':[0 for i in range(3)]}
        # represent accelX accelY accelZ gyroX gyroY gyroZ
    def read(self):
        while not self._read():
            pass
        return np.array(self.data)
    def _read(self):
        data = self.device.readline()
        data = data.split()
        if(len(data) == 4):
            self.data[str(data[0][:-1])][0]= float(data[1][:-1])
            self.data[str(data[0][:-1])][1]= float(data[2][:-1])
            self.data[str(data[0][:-1])][2]= float(data[3])
            print(self.data[str(data[0][:-1])])
            return True
        else: 
            return False

if __name__ == '__main__':
    sensor = Sensor()
