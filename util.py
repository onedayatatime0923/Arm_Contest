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
import imageio, time, math, os, random
assert os and np and F and torchvision


class Datamanager:
    def __init__(self,vocabulary_file=None, tensorboard_dir= None):
        self.voc=Vocabulary()
        self.vocabulary_file = vocabulary_file
        if tensorboard_dir== None: self.writer=None
        else: self.tb_setting(tensorboard_dir)
    def tb_setting(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        for f in os.listdir(directory): 
            os.remove('{}/{}'.format(directory,f))
        self.writer = SummaryWriter(directory)
    # for image
    def get_data_image(self,file_path,tag_path,batch_size,shuffle=True, downsample_factor=12, image_size=(224,224),save_path=None):
        if save_path is not None:
            if os.path.isfile(save_path[0]) and os.path.isfile(save_path[1]):
                x = np.load(save_path[0])
                y = np.load(save_path[1])
                self.voc.load(self.vocabulary_file)
                return DataLoader(ImageDataset(image = x,label = y, rotate = True), batch_size = batch_size, shuffle = shuffle)

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
        print('\rreading complete        '.format(f))
        x = np.array(x)
        y = np.array(y)
        if save_path is not None:
            np.save(save_path[0],x)
            np.save(save_path[1],y)
        self.voc.save(self.vocabulary_file)
        #print(x.shape)
        #print(y.shape)
        #print(y)
        #input()
        return DataLoader(ImageDataset(image = x,label = y, rotate = True), batch_size = batch_size, shuffle = shuffle)
    def get_test_data_image(self,file_path,batch_size,shuffle=True, downsample_factor=12, image_size=(224,224),save_path=None):
        if save_path is not None:
            if os.path.isfile(save_path):
                x = np.load(save_path)
                return DataLoader(ImageDataset(image = x, rotate = False), batch_size = batch_size, shuffle = shuffle)

        x = []

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
                else:
                    continue
        print('\rreading complete        '.format(f))
        x = np.array(x)
        if save_path is not None:
            np.save(save_path,x)
        #print(x.shape)
        #print(y.shape)
        #print(y)
        #input()
        return DataLoader(ImageDataset(image = x, rotate = False), batch_size = batch_size, shuffle = shuffle)
    def train_image(self, model, dataloader, epoch, optimizer, print_every= 2):
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
    def val_image(self,model,dataloader, epoch, print_every= 2):
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
    def test_image(self,model,dataloader, print_every= 2):
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
    # for rnn
    def get_data_movie(self,file_path,tag_path,batch_size,step_n, downsample_factor=12, image_size=(224,224),save_path=None):
        if save_path is not None:
            if os.path.isfile(save_path[0]) and os.path.isfile(save_path[1]):
                x = np.load(save_path[0])
                y = np.load(save_path[1])
                self.voc.load(self.vocabulary_file)
                return MovieDataLoader(image = x,label = y, step_n = step_n, batch_size = batch_size)

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
            x.append([])
            y.append([])
            for idx, im in enumerate(vid):
                if idx % downsample_factor == 0:
                    image = np.array(im).astype(np.uint8)
                    image = misc.imresize(image, size=(224,224))
                    x[-1].append(image)
                    y[-1].append(self.voc.addWord(label[f]))
                else:
                    continue
            x[-1] = np.array(x[-1])
            y[-1] = np.array(y[-1])
        print('\rreading complete        '.format(f))
        x = np.array(x)
        y = np.array(y)
        #print(x.shape)
        #print(y.shape)
        #input()
        if save_path is not None:
            np.save(save_path[0],x)
            np.save(save_path[1],y)
        self.voc.save(self.vocabulary_file)
        #print(x.shape)
        #print(y.shape)
        #print(y)
        #input()
        return MovieDataLoader(image = x,label = y, step_n = step_n, batch_size = batch_size)
    def get_test_data_movie(self,file_path,batch_size,step_n, downsample_factor=12, image_size=(224,224),save_path=None):
        if save_path is not None:
            if os.path.isfile(save_path):
                x = np.load(save_path)
                return MovieDataLoader(image = x,label = None, step_n = step_n, batch_size = batch_size)

        x = []

        file_list = [ i for i in os.listdir(file_path) if i.endswith('mp4')]
        file_list.sort()
        for f in file_list:
            vid = imageio.get_reader('{}/{}'.format(file_path,f), 'ffmpeg')
            print('\rreading from {}...'.format(f),end='')
            x.append([])
            for idx, im in enumerate(vid):
                if idx % downsample_factor == 0:
                    image = np.array(im).astype(np.uint8)
                    image = misc.imresize(image, size=(224,224))
                    x[-1].append(image)
                else:
                    continue
            x[-1] = np.array(x[-1])
        print('\rreading complete        '.format(f))
        x = np.array(x)
        if save_path is not None:
            np.save(save_path,x)
        #print(x.shape)
        #print(y.shape)
        #print(y)
        #input()
        return MovieDataLoader(image = x,label = None, step_n = step_n, batch_size = batch_size)
    def train_movie(self, model, dataloader, epoch, optimizer, print_every= 2):
        start= time.time()
        model.train()
        
        total_loss= 0
        batch_loss= 0

        total_correct= 0
        batch_correct= 0
        total_count= 0
        batch_count= 0
        
        data_size= len(dataloader)
        for b, (x, i, y, _) in enumerate(dataloader):
            batch_index=b+1
            x, i, y= Variable(x).cuda(), Variable(i).cuda(), Variable(y).cuda()
            output= model(x, i)
            loss = self.pack_CCE(output,y,i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # loss
            batch_loss+= float(loss)
            total_loss+= float(loss)* len(x)
            # accu
            correct, count = self.pack_accu(output, y, i)

            batch_correct += correct
            total_correct += correct
            batch_count += count
            total_count += count
            #print(y)
            #print(pred)
            if batch_index% print_every== 0:
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.4f}% | Time: {}  '.format(
                            epoch , batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                            batch_loss/ print_every, 100.* batch_correct/ batch_count,
                            self.timeSince(start, batch_index*len(x)/ data_size)),end='')
                batch_loss= 0
                batch_correct= 0
                batch_count= 0
        print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.4f}% | Time: {}  '.format(
                    epoch , data_size, data_size, 100.,
                    float(total_loss)/ data_size, 100.*total_correct/ total_count,
                    self.timeSince(start, 1)))
        if self.writer != None:
            self.writer.add_scalar('Train Loss', float(total_loss)/ data_size, epoch)
            self.writer.add_scalar('Train Accu',  100.*total_correct/ total_count, epoch)
        return float(total_loss)/ data_size, 100. * total_correct/ data_size
    def val_movie(self, model, dataloader, epoch, print_every= 2):
        start= time.time()
        model.eval()
        
        total_loss= 0
        batch_loss= 0

        total_correct= 0
        batch_correct= 0
        total_count= 0
        batch_count= 0
        
        data_size= len(dataloader)
        for b, (x, i, y, _) in enumerate(dataloader):
            with torch.no_grad():
                batch_index=b+1
                x, i, y= Variable(x).cuda(), Variable(i).cuda(), Variable(y).cuda()
                output= model(x, i)
                loss = self.pack_CCE(output,y,i)
                # loss
                batch_loss+= float(loss)
                total_loss+= float(loss)* len(x)
                # accu
                correct, count = self.pack_accu(output, y, i)

                batch_correct += correct
                total_correct += correct
                batch_count += count
                total_count += count
                #print(y)
                #print(pred)
                if batch_index% print_every== 0:
                    print('\rVal Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.4f}% | Time: {}  '.format(
                                epoch , batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                                batch_loss/ print_every, 100.* batch_correct/ batch_count,
                                self.timeSince(start, batch_index*len(x)/ data_size)),end='')
                    batch_loss= 0
                    batch_correct= 0
                    batch_count= 0
        print('\rVal Epoch: {} | [{}/{} ({:.0f}%)] | Loss: {:.6f} | Accu: {:.4f}% | Time: {}  '.format(
                    epoch , data_size, data_size, 100.,
                    float(total_loss)/ data_size, 100.*total_correct/ total_count,
                    self.timeSince(start, 1)))
        if self.writer != None:
            self.writer.add_scalar('Train Loss', float(total_loss)/ data_size, epoch)
            self.writer.add_scalar('Train Accu',  100.*total_correct/ total_count, epoch)
        return float(total_loss)/ data_size, 100. * total_correct/ data_size
    def test_movie(self,model,dataloader, print_every= 2):
        start= time.time()
        model.eval()
        
        data_size= len(dataloader)
        result = []
        for b, (x, i, _) in enumerate(dataloader):
            with torch.no_grad():
                batch_index=b+1
                x, i= Variable(x).cuda(), Variable(i).cuda()
                output= model(x, i)
                pred = output.data.argmax(2) # get the index of the max log-probability
                #print(pred)
                result.append(pred)
                if batch_index% print_every== 0:
                    print('\rTest | [{}/{} ({:.0f}%)] | Time: {}  '.format(
                            batch_index*len(x), data_size, 100. * batch_index*len(x)/ data_size,
                            self.timeSince(start, batch_index*len(x)/ data_size)),end='')
        print('\rTest | [{}/{} ({:.0f}%)] | Time: {}  '.format(
                    data_size, data_size, 100., self.timeSince(start, 1)))
        #result = torch.cat(result, 0)
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
    def pack_CCE(self, x, y, i):
        packed_x= nn.utils.rnn.pack_padded_sequence(x, i, batch_first=True)
        packed_y= nn.utils.rnn.pack_padded_sequence(y, i, batch_first=True)
        result = F.cross_entropy(packed_x.data,packed_y.data)
        return result
    def pack_accu(self, x, y, i):
        packed_x= nn.utils.rnn.pack_padded_sequence(x, i, batch_first=True)
        packed_y= nn.utils.rnn.pack_padded_sequence(y, i, batch_first=True)
        pred = packed_x.data.argmax(1)
        correct = int(pred.eq(packed_y.data).long().cpu().sum())
        count = len(pred)
        return correct, count
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
    def __init__(self, vocabulary_file = None):
        self.word2index= {"null":0}
        self.word2count = {}
        self.index2word = {0: "null"}
        self.n_words = 1  # Count null
        if vocabulary_file is not None:
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
        with open(path,'r') as f:
            for line in f:
                word=line.replace('\n','')
                self.addWord(word)

class Image_Classifier(nn.Module):
    def __init__(self,input_dim, label_dim, dropout=0.5):
        super(Image_Classifier, self).__init__()
        #self.cnn = torchvision.models.vgg16().features
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear( (input_dim[0]//8) * (input_dim[1]//8) * 64 , 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(256, label_dim),
        )
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
class Movie_Classifier(nn.Module):
    def __init__(self, input_dim, layer_n, label_dim, dropout = 0.5):
        super(Movie_Classifier, self).__init__()
        self.layer_n = layer_n
        hidden_dim = 256
        self.hidden= self.initHidden(hidden_dim)

        #self.cnn = torchvision.models.vgg16().features
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.rnn= nn.GRU( (input_dim[0]//16) * (input_dim[1]//16) * 128 , hidden_dim,num_layers= layer_n,batch_first=True, dropout=dropout)
        #self.rnn= nn.LSTM((input_dim[0]//8) * (input_dim[1]//8) * 64, hidden_dim,num_layers= layer_n,batch_first=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear( hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, label_dim),
        )
    def forward(self, x, i):
        #print(x.size())
        #print(i.size())
        #input()
        packed_data= nn.utils.rnn.pack_padded_sequence(x, i, batch_first=True)
        z = self.cnn(packed_data.data)
        z = z.view(z.size(0),-1)

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
        #print(z[0].size())
        #print(z[1].size())
        #input()

        return z[0]
    def hidden_layer(self,n):
        return  self.hidden.repeat(1,n,1)
    def initHidden(self, hidden_size):
        return Variable(torch.zeros(self.layer_n,1, hidden_size).cuda(),requires_grad=False)
    def save(self, path):
        torch.save(self,path)

class ImageDataset(Dataset):
    def __init__(self, image, label= None, rotate = False, angle = 10):
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
class ActionDataLoader():
    def __init__(self, image, label, batch_size, shuffle, max_len=10000 ):
        self.image = image
        self.label = label
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_len = max_len
    def __iter__(self):
        self.index = list(range(len(self.image)))
        if self.shuffle: random.shuffle(self.index)
        self.start_index=0
        self.end_index=min(len(self.image),self.start_index+self.batch_size)
        return self
    def __next__(self):
        if self.start_index >= len(self.image):
            raise StopIteration
        x,i,y=[], [], []
        for j in range(self.start_index,self.end_index):
            x.append(torch.FloatTensor(self.image[self.index[j]][:self.max_len]).permute(0,3,1,2)/255)
            i.append(min(len(self.image[self.index[j]]),self.max_len))
            if self.label is not None:
                y.append(torch.LongTensor(self.label[self.index[j]][:self.max_len]))
        sort_index= torch.LongTensor(sorted(range(len(i)), key=lambda k: i[k], reverse=True))
        sort_x=nn.utils.rnn.pad_sequence( [x[i] for i in sort_index],batch_first=True)
        sort_i= torch.index_select(torch.LongTensor(i), 0, sort_index)
        if self.label is not None:
            sort_y=nn.utils.rnn.pad_sequence( [y[i] for i in sort_index],batch_first=True)
        self.start_index+=self.batch_size
        self.end_index=min(len(self.image),self.start_index+self.batch_size)
        #print(sort_x.size())
        #print(sort_i)
        #print(sort_y)
        #input()
        if self.label is not None:
            return sort_x,sort_i,sort_y, sort_index
        elif self.label is None:
            return sort_x,sort_i,sort_index
    def __len__(self):
        return len(self.image)
    def reverse(self, x, i):
        sort_index= torch.cuda.LongTensor(sorted(range(len(i)), key=lambda k: i[k]))
        sort_x= torch.index_select(x, 0, sort_index)
        return sort_x
class MovieDataLoader():
    def __init__(self, image, label, step_n, batch_size, movie_len=3):
        self.image = image
        self.label = label
        self.step  = 0
        self.step_n  = step_n
        self.batch_size = batch_size
        self.movie_len = movie_len
    def __iter__(self):
        self.step = 0
        return self
    def __next__(self):
        if self.step >= self.step_n:
            raise StopIteration
        index = [ random.sample(range(len(self.image)),self.movie_len) for i in range(self.batch_size)]
        #print(index)
        #input()
        x,i,y=[], [], []
        for movie in index:
            movie_x = []
            movie_i = 0
            movie_y = []
            for m in movie:
                movie_x.append(torch.FloatTensor(self.image[m]).permute(0,3,1,2)/255)
                movie_i += len(self.image[m])
                if self.label is not None:
                    movie_y.append(torch.LongTensor(self.label[m]))
            x.append(torch.cat(movie_x,0))
            i.append(movie_i)
            if self.label is not None:
                y.append(torch.cat(movie_y,0))
        sort_index= torch.LongTensor(sorted(range(len(i)), key=lambda k: i[k], reverse=True))
        sort_x=nn.utils.rnn.pad_sequence( [x[i] for i in sort_index],batch_first=True)
        sort_i= torch.index_select(torch.LongTensor(i), 0, sort_index)
        if self.label is not None:
            sort_y=nn.utils.rnn.pad_sequence( [y[i] for i in sort_index],batch_first=True)

        self.step += 1
        
        #print(sort_x.size())
        #print(sort_i)
        #print(sort_y)
        #input()
        if self.label is not None:
            return sort_x,sort_i,sort_y, sort_index
        elif self.label is None:
            return sort_x,sort_i,sort_index
    def __len__(self):
        return self.batch_size * self.step_n
    def reverse(self, x, i):
        sort_index= torch.cuda.LongTensor(sorted(range(len(i)), key=lambda k: i[k]))
        sort_x= torch.index_select(x, 0, sort_index)
        return sort_x
