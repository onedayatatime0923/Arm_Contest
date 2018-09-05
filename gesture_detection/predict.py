import numpy as np
import os
from dtw import dtw
from numpy import linalg as LA

import torch
from torch.autograd import Variable
from options import RecorderOptions, MainOptions
from sensor import Sensor
from filter import Filter
from visualizer import SensorVisualizer, Painter
from models import createModel
from utils import convert
from speech import Speech

parser = MainOptions()
opt = parser.parse()
sensor = Sensor(opt.port)
sensor.flush()
filter = Filter(opt.n, opt.n)
visualizer = SensorVisualizer(repr = opt.repr)
speech = Speech()

model = createModel(opt)
model.setup(opt)
model.eval()

def reading_ref():
    action_list = []
    ref_act = []
    action = []
    with open("./utils/vocabulary/record.txt", 'r') as f:
        for line in f:
            action_list.append(line.strip())
    for act in action_list:
        if(os.path.exists("./data/{}/".format(act))):
            ref_act.append(np.array([data for data in np.load("data/{}/0.npy".format(act))[0]]))
            action.append(act)
    return action, ref_act

def predict(target, ref_act):
    score = []
    for ref in ref_act:
        score.append(dtw(ref, target, dist=lambda x, y: LA.norm(x - y, ord=1))[0])
    if min(np.array(score)) < 98.:
        print(min(np.array(score)))
        return score.index(min(np.array(score)))
    else:
        print(min(np.array(score)))
        return -1.


            


def main():
    action_list, ref_act = reading_ref()
    #print(ref_act)
    target = []
    while True:
        data = sensor.read()
        data = filter.update(data)
        x = Variable(convert(torch.FloatTensor(data)))
        signal = model.predict(x)
        if signal:
            target.append(data)
            index = predict(target, ref_act)
            if index != -1.:
                print(action_list[index])
                speech(action_list[index])
                target = []
            else:
                #print("No act")
                pass
        else:
            print("Stop")
            target = []
            pass
    
if(__name__ == '__main__'):
    main()
