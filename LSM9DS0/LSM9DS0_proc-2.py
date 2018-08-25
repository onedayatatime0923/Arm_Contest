#import pyqtgraph as pg
import pyFeature as pF
import numpy as np
import time
import random
from LSM9DS0_pyArdui import arduiData
import matplotlib.pyplot as plt
import scipy.fftpack
from matplotlib.pyplot import plot, draw, show
import pickle 

from sklearn.neural_network import MLPClassifier
#from sklearn.neighbors import KNeighborsClassifier

import socket
assert time and random and scipy and plot and draw and show and MLPClassifier

UDP_IP = "127.0.0.1"
UDP_PORT = 11111

#==============================
#      variables
#==============================

init, process, accel_begin, gyro_begin, accel_end, gyro_end = True, False, False, False, False, False
first = False
collectData = 0
DC, noise_RMS = 0, 0
begin1, begin2, end1, end2 = 0, 0, 0, 0
accelX, accelY, accelZ, gyroX, gyroY, gyroZ = 0, 0, 0, 0, 0, 0

INIT_WINDOW_INTERVAL = 20
WINDOW_INTERVAL = 1000
t = np.arange(WINDOW_INTERVAL)
init_y = np.zeros((INIT_WINDOW_INTERVAL, 6))
ay_x = np.array([])
ay_y = np.array([])
ay_z = np.array([])
gy_x = np.array([])
gy_y = np.array([])
gy_z = np.array([])
sample = np.array([])
begin_threshold = []
end_threshold = []
prev_data = np.array([])

one = []
two = []
three = []
four = []
five = []
six = []
seven = []
eight = []
nine = []
zero = []

#=====================================================
#initData: return DC offset & RMS of the first seconds
#=====================================================

def initData(data):
	accel = []
	gyro = []
	dc = np.mean(data, axis=0)
	data -= dc #DC offset
	#print(data[2])
	for i in range(INIT_WINDOW_INTERVAL):
		#print(data[i])
		#print(np.square(data[i][0:2]))
		#print(abs( np.sqrt( np.sum(np.square(data[i][0:2])) ) - 1 ))
		accel.append( abs( np.sqrt( np.sum(np.square(data[i][0:3])) ) - 1 ) )
		gyro.append( np.sqrt( np.sum(np.square(data[i][3:6])) ))
	max1 = np.amax(np.array(accel))
	max2 = np.amax(np.array(gyro))
	return dc, max1, max2

#=====================================================
#dataFeature: produce the data feature to be classified
#=====================================================

def dataFeature(data):
    mean = pF.mean(data)
    std = pF.std(data)
    #dataPoints = pF.dataPoints(data)
    highestPeakVal = pF.highestPeakVal(data)
    lowestPeakVal = pF.lowestPeakVal(data)
    highestPeaklocate = pF.highestPeaklocate(data)
    lowestPeaklocate = pF.lowestPeaklocate(data)
    peakNum = pF.peakNum(data)
    rms = pF.rms(data)
    #maxFrequency = pF.maxFrequency (data) #TODO: To modify maxFrequency
    #preRms = pF.rms(data[:80])
    #finalRms = pF.rms(data[110:230])
    #return [mean, std, highestPeakVal, lowestPeakVal, highestPeaklocate, lowestPeaklocate, peakNum, maxFrequency]#preRms,finalRms]
    tmp1 = np.hstack((mean, std))
    tmp2 = np.hstack((tmp1, highestPeakVal))
    tmp3 = np.hstack((tmp2, lowestPeakVal))
    tmp4 = np.hstack((tmp3, highestPeaklocate))
    tmp5 = np.hstack((tmp4, lowestPeaklocate))
    tmp6 = np.hstack((tmp5, rms))
    tmp7 = np.hstack((tmp6, peakNum))
    return tmp7

def cut(data,end_threshold):
  for i in range(0,980,10):
    if int(pF.rms(data[i:i+20]))<end_threshold:
      return data[:i]
  return data

def filter(data):
	d = np.array([])
	for i in range(6):
		n_iter = data.shape[1]
		sz = (n_iter,) # size of array
		Q = 1e-5 # process variance
		z = data[i]

		# allocate space for arrays
		xhat=np.zeros(sz)      # a posteri estimate of x
		P=np.zeros(sz)         # a posteri error estimate
		xhatminus=np.zeros(sz) # a priori estimate of x
		Pminus=np.zeros(sz)    # a priori error estimate
		K=np.zeros(sz)         # gain or blending factor

		R = 0.1**2 # estimate of measurement variance, change to see effect

		# intial guesses
		xhat[0] = 0.0
		P[0] = 1.0

		for k in range(1,n_iter):
		    # time update
		    xhatminus[k] = xhat[k-1]
		    Pminus[k] = P[k-1]+Q

		    # measurement update
		    K[k] = Pminus[k]/( Pminus[k]+R )
		    xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
		    P[k] = (1-K[k])*Pminus[k]
		if( i == 0 ):
			d = xhat
		else:
			d = np.vstack((d, xhat))
	return d


#==============================
#           main: :)
#==============================
while True:
	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	if( init ):   #initial 求dc / threshold
		initcount = 0
		count = 0
		while ( initcount <= INIT_WINDOW_INTERVAL + 10 ):
			accelX, accelY, accelZ, gyroX, gyroY, gyroZ = arduiData()
			
			#accelX, accelY, accelZ, gyroX, gyroY, gyroZ, pitch, roll = arduiData()
			#print(accelX, accelY, accelZ, gyroX, gyroY, gyroZ, pitch, roll)
			if( accelX != 0 and accelY != 0 and accelZ != 0 and gyroX != 0 and gyroY != 0 and gyroZ != 0):
				if( initcount > 10 ):
					'''
					try:
						print('init',accelX, accelY, accelZ, gyroX, gyroY, gyroZ)
						init_y[count][0] = accelX
						init_y[count][1] = accelY
						init_y[count][2] = accelZ
						init_y[count][3] = gyroX
						init_y[count][4] = gyroY
						init_y[count][5] = gyroZ
						count += 1
					except KeyboardInterrupt:
						print("interrupt")
						exit(-1)'''
				initcount += 1
				print(initcount)
		'''
		print(init_y)
		DC, accel_init, gyro_init = initData(init_y)
		print ('DC,accel_init,gyro_init', DC, accel_init, gyro_init)
		begin_threshold.append( 1 * accel_init )   # 1.5 * accel 的begin_threshold
		begin_threshold.append( 2 * gyro_init )    # 2 * gyro 的begin_threshold
		end_threshold = begin_threshold
		print('threshold' , begin_threshold)'''
		prev_data = np.array([ accelX, accelY, accelZ, gyroX, gyroY, gyroZ ])
		init = False
		first = True
	else:
		accelX, accelY, accelZ, gyroX, gyroY, gyroZ = arduiData()
		#accelX, accelY, accelZ, gyroX, gyroY, gyroZ, pitch, roll = arduiData()
		#raw_data = np.array([accelX, accelY, accelZ, gyroX, gyroY, gyroZ, pitch, roll])
		raw_data = np.array([accelX, accelY, accelZ, gyroX, gyroY, gyroZ])
		data = raw_data
		print(raw_data)
			
		if( accelX != 0 and accelY != 0 and accelZ != 0 and gyroX != 0 and gyroY != 0 and gyroZ != 0):
			#data[0:6] -= DC
			if(first):
				#print('prev-first', prev_data)
				difference = abs(data - prev_data)
				data = data - prev_data
				first = False
				prev_data = np.array([accelX, accelY, accelZ, gyroX, gyroY, gyroZ])
			else:
				#print('prev ', prev_data)
				difference = abs(data - prev_data)
				data = data - prev_data
				prev_data = np.array([accelX, accelY, accelZ, gyroX, gyroY, gyroZ])
			print('difference ', difference)
			print('data', data)
			#a = abs( np.sqrt( np.sum(np.square(data[0:3])) ) - 1 )
			#a = np.sqrt( np.sum(np.square(data[0:3])) )
			#b = np.sum(np.square(data[3:6]))
			if( process == False ) :
				#if( np.sqrt( np.sum(np.square(difference[0:3]))) > 5000 ):
				if( np.sqrt( np.sum(np.square(difference))) > 1000 ):
					#accel_begin = True
					begin1 += 1
					ay_x = np.append(ay_x,data[0])
					ay_y = np.append(ay_y,data[1])
					ay_z = np.append(ay_z,data[2])
					gy_x = np.append(gy_x,data[3])
					gy_y = np.append(gy_y,data[4])
					gy_z = np.append(gy_z,data[5])
					print('aaaaaaaaaa')
				else:
					ay_x = np.array([])
					ay_y = np.array([])
					ay_z = np.array([])
					gy_x = np.array([])
					gy_y = np.array([])
					gy_z = np.array([])
					begin1 = 0
				#if( np.sqrt( np.sum(np.square(difference[3:]))) > 2000 ):
				#	gyro_begin==True
				#	print('gggggggggg')

				'''
				if( np.sqrt( np.sum(np.square(data[0:3]))) > begin_threshold[0] ):
					begin1 += 1
					print('b',a)
				else:
					begin1 = 0
					print('b',a)
				
				if(np.sqrt( np.sum(np.square(data[3:6])) ) > begin_threshold[1]):
					begin2 += 1
					print('b',b)
				else:
					begin2 = 0
					print('b',b)
				
				if( begin1 > 2 ):
					accel_begin = True
				if( begin2 > 2):
					gyro_begin = True
				print('begin', begin1, begin2)'''
				#if(accel_begin==True or gyro_begin==True):
				#if(accel_begin==True):
				if( begin1 > 4 ):
					'''
					ay_x = np.append(ay_x,data[0])
					ay_y = np.append(ay_y,data[1])
					ay_z = np.append(ay_z,data[2])
					gy_x = np.append(gy_x,data[3])
					gy_y = np.append(gy_y,data[4])
					gy_z = np.append(gy_z,data[5])'''
					print("start processing...")
					process, accel_begin, gyro_begin = True, False, False
					begin1, begin2 = 0, 0

			if( process ): 
				#collectData += 1
				ay_x = np.append(ay_x,data[0])
				ay_y = np.append(ay_y,data[1])
				ay_z = np.append(ay_z,data[2])
				gy_x = np.append(gy_x,data[3])
				gy_y = np.append(gy_y,data[4])
				gy_z = np.append(gy_z,data[5])
				#if(collectData == 200):
				if(len(ay_x) == 160):
					print('datasize', collectData)
					tmp1 = np.vstack((ay_x, ay_y))
					tmp2 = np.vstack((tmp1, ay_z))
					tmp3 = np.vstack((tmp2, gy_x))
					tmp4 = np.vstack((tmp3, gy_y))
					sample = np.vstack((tmp4, gy_z))
					plt.plot(sample[0])
					sample1 = filter(sample)
					print(sample)
					#plt.plot(sample1[0])
					plt.show()
					
					#collect data
					if len(zero) != 6:
						print('===============================')
						print('zero! ',len(zero))
						print('===============================')
						#print(dataFeature(y))
						zero.append(sample1)
						one.append(sample)
					if len(zero) == 6:
						with open('data/finger_333/test22-3.p', 'wb') as fp:
							pickle.dump(zero, fp)
						with open('data/finger_333/rtest22-3.p', 'wb') as fp:
							pickle.dump(one, fp)
					ay_x = np.array([])
					ay_y = np.array([])
					ay_z = np.array([])
					gy_x = np.array([])
					gy_y = np.array([])
					gy_z = np.array([])
					process = False
					collectData = 0

					
					'''
					print('finish colllecting!!')
					# compute distances
					one = np.array([])
					two = np.array([])
					one_data = pickle.load( open( "one.p", "rb" ) )
					two_data = pickle.load( open( 'two.p', 'rb'))
					for i in range(len(one_data)):
						if( i == 0 ):
							one = dataFeature(one_data[i])
							print(one_data[i].shape)
							two = dataFeature(two_data[i])
						else:
							one = np.vstack(( one, dataFeature(one_data[i])))
							two = np.vstack(( two, dataFeature(two_data[i])))

					feature = dataFeature(sample)

					onestd = np.std(one, axis = 0)
					twostd = np.std(two, axis = 0)
					onemean = np.mean(one, axis = 0)
					twomean = np.mean(two, axis = 0)
					print('shape', one.shape)
					print('fshape', feature.shape)
					print(feature)

					featone=[0]*42
					feattwo=[0]*42
					for i in range(7*6):
						featone[i]=(2*(feature[i]-onemean[i])/(onestd[i]+twostd[i]))**2
						feattwo[i]=(2*(feature[i]-twomean[i])/(onestd[i]+twostd[i]))**2
					onenum = 0
					twonum = 0
					for i in range(7*6):
						onenum += featone[i]
						twonum += feattwo[i]
					feacompare = [ onenum,twonum ]
					finalfeat = min(range(len(feacompare)), key = lambda x: feacompare[x])
					results = ['one', 'two']  
					print( results[finalfeat], feacompare )
					MESSAGE = results[finalfeat]
					print(MESSAGE)
					#sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
					ay_x = np.array([])
					ay_y = np.array([])
					ay_z = np.array([])
					gy_x = np.array([])
					gy_y = np.array([])
					gy_z = np.array([])
					process = False
					collectData = 0
				
				if( abs( np.sqrt( np.sum(np.square(data[0:3])) ) - 1 ) < end_threshold[0] ):
					end1 += 1
					ay_x = np.append(ay_x,data[0])
					ay_y = np.append(ay_y,data[1])
					ay_z = np.append(ay_z,data[2])
					print('e', a)
					#y = np.append(y,data)
				else:
					end1 = 0
					ay_x = np.append(ay_x,data[0])
					ay_y = np.append(ay_y,data[1])
					ay_z = np.append(ay_z,data[2])
					#y = np.append(y,data)
					print('e', a)
				
				if(np.sqrt( np.sum(np.square(data[3:6])) ) < end_threshold[1]):
					end2 += 1
					gy_x = np.append(gy_x,data[3])
					gy_y = np.append(gy_y,data[4])
					gy_z = np.append(gy_z,data[5])
					print('e', b)
				else:
					end2 = 0
					gy_x = np.append(gy_x,data[3])
					gy_y = np.append(gy_y,data[4])
					gy_z = np.append(gy_z,data[5])
					print('e', b)

				if( end1 > 10):
					accel_end = True

				if(end2 > 10):
					gyro_end = True
				print('end', end1, end2)

				#if( accel_end == True or gyro_end == True):
				if( accel_end == True):
					end1, end2 = 0, 0
					accel_end, gyro_end = False, False
					print ('process ',y)
					print('datasize', collectData)
					tmp1 = np.vstack((ay_x, ay_y))
					tmp2 = np.vstack((tmp1, ay_z))
					tmp3 = np.vstack((tmp2, gy_x))
					tmp4 = np.vstack((tmp3, gy_y))
					sample = np.vstack((tmp4, gy_z))
					print(sample)
					plt.plot(ay_y)
					plt.show()

					ay_x = np.array([])
					ay_y = np.array([])
					ay_z = np.array([])
					gy_x = np.array([])
					gy_y = np.array([])
					gy_z = np.array([])
					process = False
					collectData = 0'''
					
