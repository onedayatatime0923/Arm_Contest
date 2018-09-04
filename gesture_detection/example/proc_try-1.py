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
import math
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
import tensorflow as tf
import pickle
from statistics import mean
import os
import operator
from sklearn.neural_network import MLPClassifier
#from sklearn.neighbors import KNeighborsClassifier

import socket

UDP_IP = "127.0.0.1"
UDP_PORT = 11111


segment_size = 160
num_input_channels = 6

num_training_iterations = 400
batch_size = 200

l2_reg = 5e-4
learning_rate = 5e-4 #10
dropout_rate = 0.6 #0.6
eval_iter = 1000

n_filters = 16 #10
filters_size = 16
n_hidden = 512 #1024
n_classes = 16


def weight_variable(shape, stddev):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)
  
def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv1d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_1x4(x):
    return tf.nn.max_pool(x, ksize=[1, 1, 4, 1], strides=[1, 1, 4, 1], padding='SAME')

def norm(x):
    temp = x.T - np.mean(x.T, axis = 0)
    #x = x / np.std(x, axis = 1)
    return temp.T

## Loading the dataset

print('Loading dataset...')

# Reading data
data_1=[]
data_2=[]
data_3=[]
data_4=[]
data_5=[]
data_6=[]
change_action=[0]
for i in range(n_classes):   #*****************
    if i==0:
        filename="data/finger_333/zero_200.p"
    elif i==1:
        filename="data/finger_333/one_200.p"
    elif i==2:
        filename="data/finger_333/two_200.p"
    elif i==3:
        filename="data/finger_333/three_200.p"
    elif i==4:
        filename="data/finger_333/four_200.p"
    elif i==5:
        filename="data/finger_333/five_200.p"
    elif i==6:
        filename="data/finger_333/six_200.p"
    elif i==7:
        filename="data/finger_333/seven_200.p"
    elif i==8:
        filename="data/finger_333/eight_200.p"
    elif i==9:
        filename="data/finger_333/nine_200.p"
    elif i==10:
    	filename="data/finger_333/left_200.p"
    elif i==11:
    	filename="data/finger_333/right_200.p"
    elif i==12:
    	filename="data/finger_333/del_200.p"
    elif i==13:
    	filename="data/finger_333/sp_200.p"
    elif i==14:
    	filename="data/finger_333/dot_200.p"
    elif i==15:
    	filename="data/finger_333/gg_200.p"

    objects = []
    with (open(filename, "rb")) as openfile:
        while True:
            try:
                tmp=pickle.load(openfile)
                objects.append(tmp)
            except EOFError:
                break
    for action in objects[0]:
        data_1.append(action[0])
        data_2.append(action[1])
        data_3.append(action[2])
        data_4.append(action[3])
        data_5.append(action[4])
        data_6.append(action[5])
    change_action.append(change_action[-1]+len(objects[0]))
    objects.clear()
#seperate to training_data and testing_data
'''
select=[]
for i in range(1,11):
    xs=np.arange(change_action[i-1]+1,change_action[i],1)
    xs=xs.tolist()
    select.append(random.sample(xs,1)[0])
'''
xs=np.arange(1,change_action[-1],1)
xs=xs.tolist()
test_num=int(len(data_1)*1/n_classes)   #************
select=random.sample(xs,test_num)
print(select)

#test_num=int(len(data_1)*1/10)
data_1_test=[]
data_2_test=[]
data_3_test=[]
data_4_test=[]
data_5_test=[]
data_6_test=[]

for i in select:
    data_1_test.append(data_1[i-1])
    data_2_test.append(data_2[i-1])
    data_3_test.append(data_3[i-1])
    data_4_test.append(data_4[i-1])
    data_5_test.append(data_5[i-1])
    data_6_test.append(data_6[i-1])
data_1_train = [i for j, i in enumerate(data_1) if j+1 not in select]
data_2_train = [i for j, i in enumerate(data_2) if j+1 not in select]
data_3_train = [i for j, i in enumerate(data_3) if j+1 not in select]
data_4_train = [i for j, i in enumerate(data_4) if j+1 not in select]
data_5_train = [i for j, i in enumerate(data_5) if j+1 not in select]
data_6_train = [i for j, i in enumerate(data_6) if j+1 not in select]

data_train = np.hstack((data_1_train, data_2_train, data_3_train, data_4_train, data_5_train,data_6_train))
data_test=np.hstack((data_1_test, data_2_test, data_3_test, data_4_test, data_5_test, data_6_test))
# generate basic features
features=[]
for i in range(len(data_1)):
    feature_tmp=[]
    feature_tmp.append(np.mean(data_1[i]))
    feature_tmp.append(np.mean(data_2[i]))
    feature_tmp.append(np.mean(data_3[i]))
    feature_tmp.append(np.mean(data_4[i]))
    feature_tmp.append(np.mean(data_5[i]))
    feature_tmp.append(np.mean(data_6[i]))
    feature_tmp.append(np.std(data_1[i]))
    feature_tmp.append(np.std(data_2[i]))
    feature_tmp.append(np.std(data_3[i]))
    feature_tmp.append(np.std(data_4[i]))
    feature_tmp.append(np.std(data_5[i]))
    feature_tmp.append(np.std(data_6[i]))
    features.append(feature_tmp)
features_train = [i for j, i in enumerate(features) if j+1 not in select]
features_test=[]
for i in select:
    features_test.append(features[i-1])
# generate label_vectors
labels=[]
action=1
for i in range(1,len(change_action)):
    vector=np.zeros(n_classes)      #******************
    vector=vector.tolist()
    vector[action-1]=1
    for j in range(0,change_action[i]-change_action[i-1]):
        labels.append(vector)
    action+=1
labels_train = [i for j, i in enumerate(labels) if j+1 not in select]
labels_test=[]
for i in select:
    labels_test.append(labels[i-1])
features_train = features_train - np.mean(features_train, axis = 0)
features_train = features_train / np.std(features_train, axis = 0)

features_test = features_test - np.mean(features_test, axis = 0)
features_test = features_test / np.std(features_test, axis = 0)

for i in range(num_input_channels):
    x = data_train[:, i * segment_size : (i + 1) * segment_size]
    data_train[:, i * segment_size : (i + 1) * segment_size] = norm(x)
    x = data_test[:, i * segment_size : (i + 1) * segment_size]
    data_test[:, i * segment_size : (i + 1) * segment_size] = norm(x)

train_size = data_train.shape[0]
test_size = data_test.shape[0]
num_features = features_train.shape[1]

data_test = np.reshape(data_test, [data_test.shape[0], segment_size * num_input_channels])
labels_test = np.reshape(labels_test, [len(labels_test), n_classes])
features_test = np.reshape(features_test, [features_test.shape[0], num_features])
labels_test_unary = np.argmax(labels_test, axis=1)


print("Dataset was uploaded\n")

## creating CNN

print("Creating CNN architecture\n")


# Convolutional and Pooling layers

W_conv1 = weight_variable([1, filters_size, num_input_channels, n_filters], stddev=0.01)
b_conv1 = bias_variable([n_filters])

x = tf.placeholder(tf.float32, [None, segment_size * num_input_channels])
x_image = tf.reshape(x, [-1, 1, segment_size, num_input_channels])

h_conv1 = tf.nn.relu(conv1d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_1x4(h_conv1)


# Augmenting data with statistical features

flat_size = int(math.ceil(float(segment_size)/4)) * n_filters

h_feat = tf.placeholder(tf.float32, [None, num_features])
h_flat = tf.reshape(h_pool1, [-1, flat_size])

h_hidden = tf.concat([h_flat, h_feat],1)
flat_size += num_features 

# Fully connected layer with Dropout

W_fc1 = weight_variable([flat_size, n_hidden], stddev=0.01)
b_fc1 = bias_variable([n_hidden])

h_fc1 = tf.nn.relu(tf.matmul(h_hidden, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# Softmax layer

W_softmax = weight_variable([n_hidden, n_classes], stddev=0.01)
b_softmax = bias_variable([n_classes])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_softmax) + b_softmax)
y_ = tf.placeholder(tf.float32, [None, n_classes])


# Cross entropy loss function and L2 regularization term

cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
cross_entropy += l2_reg * (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1))


# Training step

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run Tensorflow session

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train CNN
print("Training CNN... ")

max_accuracy = 0.0

for i in range(num_training_iterations):
    idx_train = np.random.randint(0, train_size, batch_size)          
    
    xt = np.reshape(data_train[idx_train], [batch_size, segment_size * num_input_channels])
    yt = np.reshape(np.array(labels_train)[idx_train], [batch_size, n_classes])
    ft = np.reshape(features_train[idx_train], [batch_size, num_features])
    sess.run(train_step, feed_dict={x: xt, y_: yt, h_feat: ft, keep_prob: dropout_rate})            
    if i % eval_iter == 0:

        train_accuracy, train_entropy, y_pred = sess.run([accuracy, cross_entropy, y_conv], 
        feed_dict={ x : data_test, y_: labels_test, h_feat: features_test, keep_prob: 1})

        if max_accuracy < train_accuracy:
            max_accuracy = train_accuracy


print(labels_test)
print("step %d, entropy %g" % (i, train_entropy))
print("step %d, max accuracy %g, accuracy %g" % (i, max_accuracy, train_accuracy))
print(classification_report(labels_test_unary, np.argmax(y_pred, axis=1), digits=4))


#==============================
#      variables
#==============================

init, process, accel_begin, gyro_begin, accel_end, gyro_end = True, False, False, False, False, False
first = False
collectData = 0
DC, noise_RMS = 0, 0
begin1, begin2, end1, end2 = 0, 0, 0, 0
NOT_TRAINED = True
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
		#print(raw_data)
			
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
				if( begin1 > 4 ):
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
					tmp1 = np.vstack((ay_x, ay_y))
					tmp2 = np.vstack((tmp1, ay_z))
					tmp3 = np.vstack((tmp2, gy_x))
					tmp4 = np.vstack((tmp3, gy_y))
					sample = np.vstack((tmp4, gy_z))
					sample = filter(sample)
					print(sample)
					#plt.plot(ay_x)
					#plt.show()
					data_sample_x=np.hstack((sample[0],sample[1], sample[2], sample[3], sample[4], sample[5]))
					data_sample_y=np.zeros(n_classes)
					data_sample_f=[]
					feature_tmp=[]
					for i in range(sample.shape[0]):
						feature_tmp.append(np.mean(sample[i]))
					for i in range(sample.shape[0]):
						feature_tmp.append(np.std(sample[i]))
					data_sample_f.append(feature_tmp)
					data_sample_f = data_sample_f - np.mean(data_sample_f, axis = 0)
					data_sample_x = np.reshape(data_sample_x, [1, segment_size * num_input_channels])
					data_sample_y= np.reshape(data_sample_y, [1, n_classes])
					data_sample_f = np.reshape(data_sample_f, [1, num_features])
					train_accuracy, train_entropy, y_pred = sess.run([accuracy, cross_entropy, y_conv], feed_dict={ x : data_sample_x, y_: data_sample_y, h_feat: data_sample_f, keep_prob: 1})
					y_total=y_pred
					for i in range(100):
						train_accuracy, train_entropy, y_pred = sess.run([accuracy, cross_entropy, y_conv], feed_dict={ x : data_sample_x, y_: data_sample_y, h_feat: data_sample_f, keep_prob: 1})
						y_total=y_total+y_pred
					for prediction in y_total:
						index, value=max(enumerate(prediction),key=operator.itemgetter(1))
						string=str(index)+" "+str(value)
						print("********")
						print(string)
					sock.sendto(str(index).encode(), (UDP_IP, UDP_PORT))

					ay_x = np.array([])
					ay_y = np.array([])
					ay_z = np.array([])
					gy_x = np.array([])
					gy_y = np.array([])
					gy_z = np.array([])
					process = False
					collectData = 0