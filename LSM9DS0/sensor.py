
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
        if(len(data.split()) == 8):
            # todo: check data content
            print('check')
            data = str(data)
            print(data)
            input()
            data = data[2:-5]
            data = data.split();
            if( data[0] == 'A'):
                self.data[0]= float(data[1])
                self.data[1]= float(data[2])
                self.data[2]= float(data[3])
            if( data[4] == 'G'):
                self.data[3]= float(data[5])
                self.data[4]= float(data[6])
                self.data[5]= float(data[7])
            return True
        else: 
            return False

if __name__ == '__main__':
    sensor = Sensor()
