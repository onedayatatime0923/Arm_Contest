
import numpy as np
import serial


class Sensor:
    def __init__(self, device = '/dev/tty0', freq = 9600, timeout = 3):
        print('------------------------------ device initializing ------------------------------')
        self.device = serial.Serial( device, freq, timeout = timeout)
        print('device {} is on {} with frequency {} Hz.'.format(self.device.name, self.device.port, self.device.baudrate))
        print('------------------------------ device initialized -------------------------------')
        self.data = [0 for i in range(6)]
        # represent accelX accelY accelZ gyroX gyroY gyroZ
    def read(self):
        while not self._read():
            pass
        return np.array(self.data)
    def _read(self):
        data = self.device.readline()
        data = data.split()
        print(len(data))
        if(len(data) == 4):
            # todo: check data content
            print('check')
            print(data)
            input()
            if( data[0] == '\rA:'):
                self.data[1]= float(data[1][:-1])
                self.data[2]= float(data[2][:-1])
                self.data[3]= float(data[3])
            if( data[0] == '\rG:'):
                self.data[5]= float(data[1][:-1])
                self.data[6]= float(data[2][:-1])
                self.data[7]= float(data[3])
            return True
        else: 
            return False

if __name__ == '__main__':
    sensor = Sensor()
