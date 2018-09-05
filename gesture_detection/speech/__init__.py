
import os

class Speech:
    def __init__(self):
        pass
    def __call__(self, data):
        os.system("google_speech -l en {}".format(data))
