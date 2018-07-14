import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import imageio, time, math, os
assert os and np and F


class Datamanager:
    def __init__(self,vocabulary_file=None, tensorboard_dir= None):
        self.voc=Vocabulary(vocabulary_file=vocabulary_file)
        if tensorboard_dir== None: self.writer=None
        else: self.tb_setting(tensorboard_dir)
    def tb_setting(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        for f in os.listdir(directory): 
            os.remove('{}/{}'.format(directory,f))
        self.writer = SummaryWriter(directory)
    def get_data(self,file_path,tag_path,batch_size,shuffle=True, downsample_factor=12, image_size=(224,224)):
        label = {}
        x = []
        y = []

        with open(tag_path,'r') as f:
            for i in f.readlines():
                data = i.strip('\n').split(',')
                label[data[0]] = data[1]

        file_list = [ i for i in os.listdir(file_path) if i.endswith('mp4')]
        file_list.sort()
        for f in file_list:
            vid = imageio.get_reader('{}/{}'.format(file_path,f), 'ffmpeg')
            print('\rreading from {}...'.format(f),end='')
            for idx, im in enumerate(vid):
                if idx % downsample_factor == 0:
                    image = np.array(im).astype(np.uint8)
                    image = misc.imresize(image, size=(224,224))
                    x.append(image)
                    y.append(self.voc.addWord(label[f]))
                else:
                    continue
        print('\rreading complete'.format(f),end='')
        x = np.array(x)
        y = np.array(y)
        #print(x.shape)
        #print(y.shape)
        #print(y)
        #input()
        return DataLoader(ImageDataset(image = x,label = y, rotate = False), batch_size = batch_size, shuffle = shuffle)
    def train_classifier(self, model, dataloader, epoch, optimizer, print_every= 2):
        start= time.time()
        model.train()
        
        criterion= nn.CrossEntropyLoss()
        total_loss= 0
        batch_loss= 0
        total_correct= 0
        batch_correct= 0
        
        data_size= len(dataloader.dataset)
        for b, (x, y) in enumerate(dataloader):
            batch_index=b+1
            x, y= Variable(x).cuda(), Variable(y).squeeze(1).cuda()
            output= model(x)
            loss = criterion(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss
            batch_loss+= float(loss)
            total_loss+= float(loss)* len(x)
            # accu
            pred = output.data.argmax(1) # get the index of the max log-probability
            #print(y)
            #print(pred)
            correct = int(pred.eq(y.data).long().cpu().sum())
            batch_correct += correct/ len(x)
            total_correct += correct
            if batch_index% print_every== 0:
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.4f}% | Time: {}  '.format(
                            epoch , batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                            batch_loss/ print_every, 100.* batch_correct/ print_every,
                            self.timeSince(start, batch_index*len(x)/ data_size)),end='')
                batch_loss= 0
                batch_correct= 0
        print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.4f}% | Time: {}  '.format(
                    epoch , data_size, data_size, 100.,
                    float(total_loss)/ data_size, 100.*total_correct/ data_size,
                    self.timeSince(start, 1)))
        if self.writer != None:
            self.writer.add_scalar('Train Loss', float(total_loss)/ data_size, epoch)
            self.writer.add_scalar('Train Accu',  100.*total_correct/ data_size, epoch)
        return float(total_loss)/ data_size, 100. * total_correct/ data_size
    def val_classifier(self,model,dataloader, epoch, print_every= 2):
        start= time.time()
        model.eval()
        
        criterion= nn.CrossEntropyLoss()
        total_loss= 0
        batch_loss= 0
        total_correct= 0
        batch_correct= 0
        
        data_size= len(dataloader.dataset)
        for b, (x, y) in enumerate(dataloader):
            with torch.no_grad():
                batch_index=b+1
                x, y= Variable(x).cuda(), Variable(y).squeeze(1).cuda()
                output= model(x)
                loss = criterion(output,y)
                # loss
                batch_loss+= float(loss)
                total_loss+= float(loss)* len(x)
                # accu
                pred = output.data.argmax(1) # get the index of the max log-probability
                correct = int(pred.eq(y.data).long().cpu().sum())
                batch_correct += correct/ len(x)
                total_correct += correct
                if batch_index% print_every== 0:
                    print('\rVal Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.4f}% | Time: {}  '.format(
                            epoch , batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                            batch_loss/ print_every, 100.* batch_correct/ print_every,
                            self.timeSince(start, batch_index*len(x)/ data_size)),end='')
                    batch_loss= 0
                    batch_correct= 0
        print('\rVal Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.4f}% | Time: {}  '.format(
                    epoch , data_size, data_size, 100.,
                    float(total_loss)/ data_size, 100.*total_correct/ data_size,
                    self.timeSince(start, 1)))
        if self.writer != None:
            self.writer.add_scalar('Val Loss', float(total_loss)/ data_size, epoch)
            self.writer.add_scalar('Val Accu',  100.*total_correct/ data_size, epoch)
        return float(total_loss)/ data_size, 100. * total_correct/ data_size
    def test_classifier(self,model,dataloader, print_every= 2):
        start= time.time()
        model.eval()
        
        data_size= len(dataloader.dataset)
        result = []
        for b, x in enumerate(dataloader):
            with torch.no_grad():
                batch_index=b+1
                x = Variable(x).cuda()
                output= model(x)
                pred = output.data.argmax(1) # get the index of the max log-probability
                result.append(pred)
                if batch_index% print_every== 0:
                    print('\rTest | [{}/{} ({:.0f}%)] | Time: {}  '.format(
                            batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                            self.timeSince(start, batch_index*len(x)/ data_size)),end='')
        print('\rTest | [{}/{} ({:.0f}%)] | Time: {}  '.format(
                    data_size, data_size, 100., self.timeSince(start, 1)))
        result = torch.cat(result, 0)
        return result
    def timeSince(self,since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.asMinutes(s), self.asMinutes(rs))
    def asMinutes(self,s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)
    def write(self,path,decoded_words,name,video):
        with open(path,'w') as f:
            for i in range(len(video)):
                seq_list=[]
                for j in decoded_words[i]:
                    index=int(j)
                    if index ==self.voc.word2index('EOS'): break
                    seq_list.append(self.voc.index2word[index])
                d_seq=' '.join(seq_list)
                f.write('{},{}\n'.format(video[i],d_seq))
    def count_parameters(self,model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    def plot(self, record, path):
        x=np.array(list(range(1,len(record)+1)),dtype=np.uint8)
        y=np.array(record)
        plt.figure()
        plt.plot(x,y[:,0],'b',label='loss')
        plt.plot(x,y[:,1],'g',label='bleu')
        plt.legend()
        plt.savefig(path)
        #plt.close()

class Vocabulary:
    def __init__(self, vocabulary_file):
        if vocabulary_file == None:
            self.word2index= {"null":0}
            self.word2count = {}
            self.index2word = {0: "null"}
            self.n_words = 1  # Count null
        else:
            self.load(vocabulary_file)
    def addWord(self, word):
        word=word.lower()
        if word in self.word2count: self.word2count[word]+=1
        else: 
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        return self.word2index[word]
    def save(self, path):
        index_list= sorted( self.word2index , key= self.word2index.get)
        with open( path, 'w') as f:
            f.write('\n'.join(index_list))
    def load(self, path):
        self.w2i= {}
        self.word2count= {}
        self.index2word= {}
        with open(path,'r') as f:
            i=0
            for line in f:
                word=line.replace('\n','')
                self.word2index[word] = i
                self.word2count[word]=0
                self.index2word[i] = word
                i+=1
            self.n_words=len(self.w2i)

class Classifier(nn.Module):
    def __init__(self,input_dim, label_dim, dropout=0.5):
        super(Classifier, self).__init__()
        self.cnn = torchvision.models.vgg16().features
        self.classifier = nn.Sequential(
            nn.Linear( (input_dim[0]//32) * (input_dim[1]//32) * 512 , 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, label_dim),
        )
    def forward(self, x):
        x = self.cnn(x)
        return x
class Rnn_Classifier_Movie(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_n, label_dim, dropout ):
        super(Rnn_Classifier_Movie, self).__init__()
        self.layer_n = layer_n
        self.hidden= self.initHidden(hidden_dim)

        self.cnn = torchvision.models.vgg16.features
        self.rnn= nn.GRU( input_dim, hidden_dim,num_layers= layer_n,batch_first=True, dropout=dropout)
        #self.rnn= nn.LSTM( input_dim, hidden_dim,num_layers= layer_n,batch_first=True, dropout=dropout)
        self.classifier = nn.Sequential(
                nn.Linear( hidden_dim ,hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                #nn.SELU(),
                nn.Linear( hidden_dim,hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                #nn.SELU(),
                nn.Dropout(dropout),
                nn.Linear( hidden_dim,hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                #nn.SELU(),
                nn.Dropout(dropout),
                nn.Linear( hidden_dim,label_dim))
    def forward(self, x, i):
        packed_data= nn.utils.rnn.pack_padded_sequence(x, i, batch_first=True)
        z = self.cnn(packed_data.data)

        packed_data= nn.utils.rnn.PackedSequence( z, packed_data.batch_sizes)
        packed_data, hidden=self.rnn(packed_data, self.hidden_layer(len(x)))

        z = self.classifier(packed_data.data)

        packed_data= nn.utils.rnn.PackedSequence( z, packed_data.batch_sizes)
        z = nn.utils.rnn.pad_packed_sequence(packed_data,batch_first=True)

        #z = hidden.permute(1,0,2).contiguous().view(hidden.size(1),-1)
        #z=torch.mean(torch.transpose(hidden,0,1).contiguous(),1)


        #index= i.unsqueeze(1).unsqueeze(2).repeat(1,1,z[0].size(2))
        #z= torch.gather(z[0],1,index-1).squeeze(1)

        #z=torch.sum(z[0],1)/ i.float().unsqueeze(1).repeat(1,z[0].size(2))

        return z[0]
    def hidden_layer(self,n):
        return  self.hidden.repeat(1,n,1)
    def initHidden(self, hidden_size):
        return Variable(torch.zeros(self.layer_n,1, hidden_size).cuda())#,requires_grad=True)
    def save(self, path):
        torch.save(self,path)

class ImageDataset(Dataset):
    def __init__(self, image, label= None, rotate = False, angle = 5):
        self.image = image
        self.label = label

        self.rotate= rotate
        self.transform_rotate= transforms.Compose([transforms.ToPILImage(), transforms.RandomRotation(angle),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform_norotate= transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.angle= angle
    def __getitem__(self, i):
        x= self.image[i]
        if self.rotate: x= self.transform_rotate(x)
        else: x= self.transform_norotate(x)

        if self.label is not None:
            y=torch.LongTensor([self.label[i]])
            return x,y
        else :
            return x
    def __len__(self):
        return len(self.image)
