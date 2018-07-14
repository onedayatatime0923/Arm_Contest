import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from torch import optim
import torchvision
import os
import numpy as np
import json
import random
import time
import math
import matplotlib.pyplot as plt
assert os and np and F


class Datamanager:
    def __init__(self,vocabulary_file=None, min_count= None, max_length=0):
        self.voc=Vocabulary(vocabulary_file=vocabulary_file,min_count=min_count)
        self.data={}
        self.vocab_size= self.voc.n_words
        self.max_length=max_length
    def get_data(self,name,f_path,l_path,mode,batch_size,shuffle=True):
        # self.data[name]=[ dataloader, labels]
        feats={}
        captions_id={}
        captions_str={}
        max_sen=0
        for i in os.listdir(f_path):
            if not i.startswith('.'):
                x=torch.FloatTensor(np.load('{}/{}'.format(f_path,i)))
                feats[i[:-4]]=x

        with open(l_path) as f:
            labels=json.load(f)
        if mode== 'train':
            for l in labels:
                m=self.voc.addSentence(l['caption'])
                if m>max_sen: max_sen=m
            self.max_length=max_sen+2
            self.vocab_size=self.voc.n_words
            # save the captions_str is for getting the grounded sequence when evaluating
            for l in labels:
                captions_id[l['id']]=self.IndexFromSentence(l['caption'],begin=True,end=True)
                captions_str[l['id']]=[x.rstrip('.') for x in l['caption']]
        elif mode== 'test':
            # save the captions_str is for getting the grounded sequence when evaluating
            for l in labels:
                captions_id[l['id']]=self.IndexFromSentence([l['caption'][0]],begin=True,end=True)
                captions_str[l['id']]= [x.rstrip('.') for x in l['caption']]
        else : raise ValueError('Wrong mode.')
        dataset=VideoDataset(feats,captions_id)
        self.data[name]= [DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), captions_str]
    def get_test_data(self,name,f_path, batch_size,shuffle=False):
        # self.data[name]=dataloader
        feats={}
        for i in os.listdir(f_path):
            if not i.startswith('.'):
                x=torch.FloatTensor(np.load('{}/{}'.format(f_path,i)))
                feats[i[:-4]]=x

        dataset=VideoDataset(feats)
        self.data[name]= DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    def IndexFromSentence(self,sentences,begin=False,end=True):
        indexes=[]
        for s in sentences:
            index=[]
            if begin: index.append(self.voc.word2index('SOS'))
            index.extend([self.voc.word2index(word) for  word in s.split(' ')])
            if end: index.append(self.voc.word2index('EOS'))
            if len(index)< self.max_length : 
                index.extend([self.voc.word2index('PAD') for i in range(self.max_length  -len(index))])
            indexes.append(index)
        indexes = torch.LongTensor(indexes)
        return indexes
    def train(self,input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, words, teacher_forcing_ratio=1):
        encoder.train()
        decoder.train()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss = Variable(torch.cuda.FloatTensor([0]).cuda())
        loss_n = 0

        encoder_outputs, encoder_hidden = encoder(input_variable)


        #decoder_hidden= decoder.hidden_layer(len(input_variable))
        decoder_hidden= encoder_hidden
        decoder_input=torch.index_select(target_variable, 1, Variable(torch.LongTensor([0])).cuda())
        
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        for di in range(1,self.max_length):
            decoder_output, decoder_hidden= decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            if use_teacher_forcing:
                # Teacher forcing: Feed the target as the next input
                decoder_input=torch.index_select(target_variable, 1, Variable(torch.LongTensor([di])).cuda())
                target=decoder_input.view(-1)
                l,n = self.loss(criterion,decoder_output, target)
                loss = loss + l
                loss_n += n
                words.append(decoder_output.data.max(1,keepdim=True)[1])
            else:
                # Without teacher forcing: use its own predictions as the next input
                ni = decoder_output.data.max(1,keepdim=True)[1]
                decoder_input = Variable(ni)
                target=torch.index_select(target_variable, 1, Variable(torch.LongTensor([di])).cuda()).view(-1)
                l,n = self.loss(criterion,decoder_output, target)
                loss = loss + l
                loss_n += n
                words.append(ni)

        loss=loss / loss_n
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        return float(loss)/ (self.max_length)
    def trainIters(self,encoder, decoder, name, test_name, n_epochs, write_file, plot_file, learning_rate=0.001, print_every=2):
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
        
        criterion = nn.CrossEntropyLoss(size_average=False)
        teacher_forcing_ratio=F.sigmoid(torch.linspace(30,-5,n_epochs))
        data_size = len(self.data[name][0].dataset)
        record=0
        loss_bleu_list=[]
        for epoch in range(n_epochs):
            start = time.time()
            loss_total=0
            print_loss_total = 0  # Reset every print_every
            bleu=[]
            for step, (batch_x, batch_y, video) in enumerate(self.data[name][0]):
                batch_index=step+1
                batch_x=Variable(batch_x).cuda()
                batch_y=Variable(batch_y).cuda()
                words=[]

                loss = self.train(batch_x, batch_y, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, words, teacher_forcing_ratio=teacher_forcing_ratio[epoch])
                # loss
                loss_total+=loss
                # bleu
                words= torch.cat(words,1)
                bleu.extend(self.bleu_batch(words,name,video[0]))

                if batch_index% print_every == 0:
                    print_loss_avg = (loss_total - print_loss_total )/ print_every
                    print_loss_total = loss_total
                    print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)] |  Loss: {:.6f} | Time: {}  '.format(
                                epoch+1 , batch_index*len(batch_x), data_size,
                                100. * batch_index*len(batch_x)/ data_size, print_loss_avg,
                                self.timeSince(start, batch_index*len(batch_x)/ data_size)),end='')
            bleu_average = sum(bleu) / len(bleu)
            print('\nTime: {} | Total loss: {:.4f} | Bleu Score: {:.5f}'.format(self.timeSince(start,1),loss_total/batch_index,bleu_average))
            print('-'*80)
            if (epoch+1)%10==0: self.evaluate(encoder,decoder,name, n=3)
            record = self.evaluate(encoder,decoder,test_name, write_file=write_file,record= record, n=5)
            loss_bleu_list.append([loss_total/ batch_index, bleu_average])
            if plot_file != None: self.plot(loss_bleu_list, plot_file)
    def evaluate(self,encoder, decoder, name, write_file=None, record=0, n=5):
        encoder.eval()
        decoder.eval()

        start = time.time()
        loss=0
        loss_n=0
        decoded_words = []
        videos = [[],[]]
        criterion = nn.CrossEntropyLoss(size_average=False)

        print_image=[random.choice(list(self.data[name][0].dataset.feats.keys())) for i in  range(n)]

        data_size = len(self.data[name][0].dataset)
        for step, (batch_x, batch_y,video) in enumerate(self.data[name][0]):
            batch_index=step+1
            batch_x=Variable(batch_x).cuda()
            batch_y=Variable(batch_y).cuda()

            encoder_outputs, encoder_hidden = encoder(batch_x)

            decoder_hidden= encoder_hidden
            decoder_input=torch.index_select(batch_y, 1, Variable(torch.LongTensor([0])).cuda())

            words=[]
            bleu=[]

            for di in range(1,self.max_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                ni = decoder_output.data.max(1,keepdim=True)[1]
                decoder_input = Variable(ni)
                words.append(ni)

                target=torch.index_select(batch_y, 1, Variable(torch.LongTensor([di])).cuda()).view(-1)
                l, n = self.loss(criterion,decoder_output, target)
                loss += float(l)
                loss_n += n

            words= torch.cat(words,1)
            bleu.extend(self.bleu_batch(words, name, video[0]))

            decoded_words.extend(words.unsqueeze(1))
            videos[0].extend(video[0])
            videos[1].extend(video[1])

            loss /= loss_n

            print('\r{} | [{}/{} ({:.0f}%)] |  Loss: {:.6f} | Time: {}  '.format(
                        name.upper(),
                        batch_index*len(batch_x), data_size,
                        100. * batch_index*len(batch_x)/ data_size, loss,
                        self.timeSince(start, batch_index*len(batch_x)/ data_size)),end='')

        bleu_average = sum(bleu) / len(bleu)
        decoded_words=torch.cat(decoded_words,0)
        print('\nTime: {} | Bleu Score: {:.5f}'.format(self.timeSince(start,1),bleu_average))
        # output decoded and ground sequence
        for i in print_image:
            seq_id=videos[0].index(i)
            seq_list=[]
            for j in decoded_words[seq_id]:
                index=int(j)
                if index ==self.voc.word2index('EOS'): break
                seq_list.append(self.voc.index2word[index])
            d_seq=' '.join(seq_list)
            g_seq=self.data[name][1][i][videos[1][seq_id]]
            print('id: {:<25} | decoded_sequence: {}'.format(i,d_seq))
            print('    {:<25} | ground_sequence: {}'.format(' '*len(i),g_seq))
        # writing output file
        if write_file!=None and bleu_average > record:
            self.write(write_file,decoded_words,name,videos[0])
            torch.save(encoder,'encoder.pt')
            torch.save(decoder,'decoder.pt')
        print('-'*80)
        return max(bleu_average, record)
    def predict(self,encoder, decoder, name, write_file=None):
        encoder.eval()
        decoder.eval()

        start = time.time()
        decoded_words = []
        videos = [[],[]]

        data_size = len(self.data[name].dataset)
        for step, (batch_x,video) in enumerate(self.data[name]):
            batch_index=step+1
            batch_x=Variable(batch_x).cuda()

            encoder_outputs, encoder_hidden = encoder(batch_x)

            decoder_hidden= encoder_hidden
            decoder_input = Variable(torch.LongTensor([self.voc.word2index('SOS') for i in range(len(batch_x))]).cuda())       

            words=[]

            for di in range(1,self.max_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                ni = decoder_output.data.max(1,keepdim=True)[1]
                decoder_input = Variable(ni)
                words.append(ni)

            words= torch.cat(words,1)
            decoded_words.extend(words.unsqueeze(1))
            videos[0].extend(video[0])
            videos[1].extend(video[1])


            print('\r{} | [{}/{} ({:.0f}%)] | Time: {}  '.format(
                        name.upper(),
                        batch_index*len(batch_x), data_size,
                        100. * batch_index*len(batch_x)/ data_size,
                        self.timeSince(start, batch_index*len(batch_x)/ data_size)),end='')

        decoded_words=torch.cat(decoded_words,0)
        print('\nTime: {}  '.format(self.timeSince(start,1)))
        # writing output file
        if write_file!=None:
            self.write(write_file,decoded_words,name,videos[0])
        print('-'*80)
    def loss(self,criterion,output,target):
        check_t=(target!=self.voc.word2index("PAD"))
        t=torch.masked_select(target,check_t).view(-1)
        check_o=check_t.view(-1,1)
        o=torch.masked_select(output,check_o).view(-1,self.vocab_size)
        if len(t)==0: return 0,0
        else : return criterion(o,t),len(t)
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
    def bleu_batch(self, words, name, video):
        bleu=[]
        for i in range(len(video)):
            seq_list=[]
            for j in words[i]:
                index=int(j)
                if index ==self.voc.word2index('EOS'): break
                seq_list.append(self.voc.index2word[index])
            seq_list=' '.join(seq_list)
            target_list=self.data[name][1][video[i]]
            if (len(seq_list)!=0):
                bleu.append(self.BLEU(seq_list,target_list,True))
        return bleu
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
    def save(self, path):
        index_list= sorted( self.w2i , key= self.w2i.get)
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
    def __init__(self,input_size, hidden_size, layer_n, dropout=0.3):
        super(Classifier, self).__init__()
        self.cnn = torchvision.models.vgg16.features
        self.hidden = self.initHidden(layer_n)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers= layer_n, batch_first=True, dropout=dropout)
    def forward(self, x):
        hidden = torch.cat([self.hidden for i in range(len(x))],1)
        output, hidden = self.rnn(x, hidden)
        output = output / torch.matmul(torch.norm(output,2,dim=2).unsqueeze(2),Variable(torch.ones(1,self.hidden_size)).cuda())
        return output,  hidden
    def initHidden(self,layer_n):
        return Variable(torch.zeros(layer_n,1, self.hidden_size),requires_grad=True).cuda()
class Rnn_Classifier_Movie(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_n, label_dim, dropout ):
        super(Rnn_Classifier_Movie, self).__init__()
        self.layer_n = layer_n
        self.hidden= self.initHidden(hidden_dim)

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

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size, layer_n, hop_n, dropout):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size 
        self.hop_n= hop_n 
        self.hidden= self.initHidden(layer_n)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.attn = nn.Sequential( nn.Linear(self.hidden_size * (layer_n+1), self.hidden_size),
                    nn.SELU(),
                    nn.Dropout(dropout))
        self.attn_weight = nn.Softmax(1)
        self.attn_combine = nn.Sequential( nn.Linear(self.hidden_size * 2, self.hidden_size))
        self.rnn= nn.GRU(self.hidden_size, self.hidden_size,num_layers= layer_n,batch_first=True, dropout=dropout)
        self.out = nn.Sequential( nn.Linear(self.hidden_size, self.vocab_size),
                    nn.SELU(),
                    nn.Dropout(dropout))
    def forward(self, x, hidden, encoder_outputs):
        # x size: batch * 1
        # encoder outputs size: batch * 80 * hidden
        # hidden size: 1 * batch * hidden
        embedded = self.embedding(x).squeeze(1)         # batch *  hidden

        h=torch.transpose(hidden,0,1).contiguous().view(hidden.size()[1],-1)
        z = self.attn(torch.cat((embedded, h), 1)) # batch * hidden
        # hopping
        for n in range(self.hop_n):
            weight = self.attn_weight(torch.bmm(encoder_outputs,z.unsqueeze(2)).squeeze(2)) # batch * 80 
            z = torch.bmm(weight.unsqueeze(1),encoder_outputs).squeeze(1) # batch * hidden

        output = self.attn_combine(torch.cat((embedded, z), 1)).unsqueeze(1)

        output, hidden=self.rnn(output, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden
    def hidden_layer(self,n):
        return  torch.cat([self.hidden for i in range(n)],1)
    def initHidden(self,layer_n):
        return Variable(torch.zeros(layer_n,1, self.hidden_size),requires_grad=True).cuda()
class VideoDataset(Dataset):
    def __init__(self, feats, captions=None):
        self.feats=feats
        self.captions=captions
        index=[]
        if captions != None:
            for i in captions:
                index.extend([(i,j) for j in range(len(captions[i]))])
        else :
            index.extend([(i,0) for i in feats])
        self.index=index
        #print(len(index))
    def __getitem__(self, i):
        x=self.feats[self.index[i][0]]
        #x+=torch.normal(torch.zeros_like(x),0.1)
        if self.captions != None:
            y=self.captions[self.index[i][0]][self.index[i][1]]
            return x,y,self.index[i]
        else :
            return x,self.index[i]
    def __len__(self):
        return len(self.index)
