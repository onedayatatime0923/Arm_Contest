
import numpy as np
import serial
import collections


class Sensor:
    def __init__(self, device = '/dev/tty0', freq = 9600, timeout = 3):
        print('------------------------------ device initializing ------------------------------')
        self.device = serial.Serial( device, freq, timeout = timeout)
        print('device {} is on {} with frequency {} Hz.'.format(self.device.name, self.device.port, self.device.baudrate))
        print('------------------------------ device initialized -------------------------------')
        self.data = collections.OrderedDict()
        self.data['A'] = [0 for i in range(3)]
        self.data['G'] = [0 for i in range(3)]
        self.data['M'] = [0 for i in range(3)]
        self.flag = collections.OrderedDict()
        self.flag['A'] = False
        self.flag['G'] = False
        self.flag['M'] = False
    def read(self):
        # return ( type : np.array, shape: (9,1))
        # meaning accelX accelY accelZ gyroX gyroY gyroZ magneticX magneticY magneticZ define in opt
        while not all(value == True for value in self.flag.values()):
            self._read()
        for key in self.flag:
            self.flag[key] = False
        data = self._convert(self.data)
        return data

    def _read(self):
        data = self.device.readline()
        data = data.split()
        if(len(data) == 4):
            self.data[(data[0][:-1]).decode('ascii')][0]= float(data[1][:-1])
            self.data[(data[0][:-1]).decode('ascii')][1]= float(data[2][:-1])
            self.data[(data[0][:-1]).decode('ascii')][2]= float(data[3])
            self.flag[(data[0][:-1]).decode('ascii')]= True

    def _convert(self,data):
        data = np.expand_dims(np.concatenate(list(data.values()), 0),1)
        return data

if __name__ == '__main__':
    sensor = Sensor()
