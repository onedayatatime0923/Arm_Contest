
import os
from dtw import dtw

class Classifier:
    def read():
        action_list = []
        ref_act = []
        action = {}
        with open("./utils/vocabulary/record.txt", 'r') as f:
            for line in f:
                act = line.strip()
                path = os.path.join('data',act)
                fileList = os.listdir(path) if os.path.exists(path) else []
                if len(fileList) > 0:

                if(os.path.exists("./data/{}/".format(act))):
                    ref_act.append(np.array([data for data in np.load("data/{}/0.npy".format(act))[0]]))
                action_list.append(line.strip())
        for act in action_list:
            if(os.path.exists("./data/{}/".format(act))):
                ref_act.append(np.array([data for data in np.load("data/{}/0.npy".format(act))[0]]))
                action.append(act)
        return action, ref_act
