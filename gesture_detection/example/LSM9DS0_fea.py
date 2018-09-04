import numpy as np
from scipy import signal, fftpack
import pickle 
import random as rd
import matplotlib.pyplot as plt

def rms(l):
    ll = []
    for i in range(l.shape[0]):
        e = np.sqrt(np.mean(np.square(l[i])))
        ll.append(e)
    return ll

def maxFrequency(A):
    #A=[]

    N = len(A)
    # sample spacing
    T = 2.273 / 1000.0 #sample rate
    x = np.linspace(0.0, N*T, N)
    #y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)

    y=A
    yf = fftpack.fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
    #print(xf)
    #print(xf,1.0/(2.0*T)/(N/2-1),len(xf))


    #fig, ax = plt.subplots()
    #ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    #plt.show()
    #print(A)
    value, interval = abs(yf), 1.0/(2.0*T)/(N/2-1) #value, inteval
    return highestPeaklocate(value) * interval
    
def mean(l):
    ll = []
    for i in range(l.shape[0]):
        e = np.mean(l[i])
        ll.append(e)
    return ll

def std(l):
    ll = []
    for i in range(l.shape[0]):
        e = np.std(l[i])
        ll.append(e)
    return ll

def dataPoints(l):
    l = np.array(l)
    return np.size(l)

def highestPeakVal(l):
    ll = []
    for i in range(l.shape[0]):
        e = np.amax(l[i])
        ll.append(e)
    return ll

def lowestPeakVal(l):
    ll = []
    for i in range(l.shape[0]):
        e = np.amin(l[i])
        ll.append(e)
    return ll

def highestPeaklocate(l):
    ll = []
    for i in range(l.shape[0]):
        e = np.argmax(l[i])
        ll.append(e)
    return ll

def lowestPeaklocate(l):
    ll = []
    for i in range(l.shape[0]):
        e = np.argmin(l[i])
        ll.append(e)
    return ll

def peakNum(l):
    '''
    indexes = signal.find_peaks_cwt(l, np.arange(5, 10))
    indexed = np.array(indexes) - 1
    return np.size(indexed)
    '''
    ll = []
    for j in range(l.shape[0]):
        num=0
        for i in range(3,len(l[j])-4,1):
            if l[j][i]>l[j][i-1] and l[j][i]>l[j][i-2] and l[j][i]>l[j][i-3] and l[j][i]>l[j][i+1] and l[j][i]>l[j][i+2] and l[j][i]>l[j][i+3] and l[j][i]>100:
                num+=1
            elif l[j][i]<l[j][i-1] and l[j][i]<l[j][i-2] and l[j][i]<l[j][i-3] and l[j][i]<l[j][i+1] and l[j][i]<l[j][i+2] and l[j][i]<l[j][i+3] and l[j][i]<-100:
                num+=1
        ll.append(num)
    return ll

if __name__ == '__main__':    
    #vec = [rd.randint(-2500,2500) for i in xrange(3000)]
    data = pickle.load( open( "one.p", "rb" ) )
    plt.plot(data[0][0])
    plt.show()
    for i in range(len(data)):
        print(mean(data[i]))
        if( i == 0 ):
            mean = mean(data[0])
            std = std(data[0])
            highestPeakVal = highestPeakVal(data[i])
            lowestPeakVal = lowestPeakVal(data[i])
            highestPeaklocate = highestPeaklocate(data[i])
            lowestPeaklocate = lowestPeaklocate(data[i])
            peakNum = peakNum(data[i])
            rms = rms(data[i])
            tmp1 = np.hstack((mean, std))
            tmp2 = np.hstack((tmp1, highestPeakVal))
            tmp3 = np.hstack((tmp2, lowestPeakVal))
            tmp4 = np.hstack((tmp3, highestPeaklocate))
            tmp5 = np.hstack((tmp4, lowestPeaklocate))
            tmp6 = np.hstack((tmp5, rms))
            tmp7 = np.hstack((tmp6, peakNum))
            one = tmp7
            print(tmp7)
        else:
            mean = mean(data[i])
            std = std(data[i])
            highestPeakVal = highestPeakVal(data[i])
            lowestPeakVal = lowestPeakVal(data[i])
            highestPeaklocate = highestPeaklocate(data[i])
            lowestPeaklocate = lowestPeaklocate(data[i])
            peakNum = peakNum(data[i])
            rms = rms(data[i])
            tmp1 = np.hstack((mean, std))
            tmp2 = np.hstack((tmp1, highestPeakVal))
            tmp3 = np.hstack((tmp2, lowestPeakVal))
            tmp4 = np.hstack((tmp3, highestPeaklocate))
            tmp5 = np.hstack((tmp4, lowestPeaklocate))
            tmp6 = np.hstack((tmp5, rms))
            tmp7 = np.hstack((tmp6, peakNum))
            print(tmp7)
            one = np.vstack(( one, tmp7))
    print(one)
    '''
    print(one_data[0])
    print(rms(one_data[0]))
    print(mean(one_data[0]))
    print(std(one_data[0]))
    #print(dataPoints(one_data[0]))
    print(highestPeakVal(one_data[0]))
    print(highestPeaklocate(one_data[0]))
    print(peakNum(one_data[0]))'''
    #print(maxFrequency(one_data[0]))
