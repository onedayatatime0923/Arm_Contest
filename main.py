
from util import Datamanager, Classifier
import torch
assert torch

#PRINT_OUTPUT_PATH = './record.png'
#WRITE_OUTPUT_PATH = './data/output.txt'
EPOCH= 50
BATCH_SIZE= 128
IMAGE_SIZE = (224,224)
LEARNING_RATE = 1E-3
DROPOUT= 0.5
VOLCABULARY_PATH = './vocab.txt'
TENSORBOARD_DIR = './runs'


dm = Datamanager(tensorboard_dir = TENSORBOARD_DIR)
train_dataloader = dm.get_data('./data/movie/','./data/tag.txt',batch_size=BATCH_SIZE, shuffle = True, image_size = IMAGE_SIZE)
#dm.get_data('test','./data/testing_data/feat','./data/testing_label.json','test',batch_size=BATCH_SIZE,shuffle=False)

dm.voc.save(VOLCABULARY_PATH)

model= Classifier(IMAGE_SIZE, dm.voc.n_words, DROPOUT)
print('Model parameters: {}'.format(dm.count_parameters(model)))

optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

accu_record=0
for epoch in range(1,EPOCH+1):
    dm.train_classifier( model, train_dataloader, epoch, optimizer)
    #record=dm.val_classifier( model, val_dataloader, epoch)
    '''
    if record[1]> accu_record:
        torch.save(model,OUTPUT_PATH)
        accu_record= record[1]
        print('Model saved!!!')
    '''
    print('='*80)
