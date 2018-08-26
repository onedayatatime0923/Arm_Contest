
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
assert Q_discrete_white_noise


class Filter:
    def __init__(self, dimState, dimObserve, covariance = 1, processNoise = 0.01, measureNoise = 0.01, dT = 0.1 ):
        self.dimState = dimState
        self.dimObserve = dimObserve
        self.covariance = covariance
        self.processNoise = processNoise
        self.measureNoise = measureNoise
        self.dT = dT # not use now
        self.reset()

    def reset(self):
        self.filter = KalmanFilter(dim_x= self.dimState, dim_z= self.dimObserve)
        # set initial state
        self.filter.x = np.array([[0] for i in range(self.dimState)],dtype = np.float)
        # set initial covariance matrix
        self.filter.P = np.eye(self.dimState) * self.covariance
        # set process noise
        self.filter.Q = np.eye(self.dimState) * self.processNoise
        #self.filter.Q = Q_discrete_white_noise(dim= self.dimState, dt= self.dT, var= self.processNoise)
        # set measure noise
        self.filter.R = np.eye(self.dimState) * self.measureNoise
        # set measure function
        self.filter.H = np.eye(self.dimObserve, self.dimState)
        # set state transition matrix:
        self.filter.F = np.eye(self.dimState)

    def update(self, data):
        self.filter.predict()
        self.filter.update(data)
        return self.filter.x
    
        
if __name__ == '__main__':
    f = Filter( 6, 6)

