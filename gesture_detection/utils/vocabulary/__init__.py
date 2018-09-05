
import os.path

class Vocabulary:
    def __init__(self):
        self.filePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'record.txt')
        self.word2index= {}
        self.index2word = {}
        self.n_words = 0
        self.load()
    def add(self, word):
        word=word.lower()
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
        self.save()
        return self.word2index[word]
    def save(self):
        index_list= sorted( self.word2index, key= self.word2index.get)
        with open( self.filePath, 'w') as f:
            f.write('\n'.join(index_list))
    def load(self):
        with open(self.filePath,'r') as f:
            i=0
            for line in f:
                word=line.replace('\n','')
                self.word2index[word] = i
                self.index2word[i] = word
                i+=1
        self.n_words=len(self.word2index)
