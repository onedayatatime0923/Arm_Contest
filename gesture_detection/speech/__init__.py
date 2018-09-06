
import os
import multiprocessing as mp

class Speech:
    def __init__(self):
        pass
    def _speech(self, data):
        os.system("google_speech -l en {}".format(data))
    def __call__(self, data):
        p = mp.Process(target=self._speech, args=(data,))
        p.start()
