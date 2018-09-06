import numpy as np

action = [line.strip() for line in open("./utils/vocabulary/record.txt", 'r')]

def dump(act):
    for i in range(20):
        data = np.load("data/{}/{}.npy".format(act, i))[0]
        label = np.full((data.shape[0],1), action.index(act))
        np.save("data/{}/{}.npy".format(act,i), [data, label])
dump("I")
