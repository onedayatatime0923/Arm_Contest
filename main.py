
from util import Datamanager, Image_Classifier, Movie_Classifier
import torch
assert torch and Image_Classifier

#PRINT_OUTPUT_PATH = './record.png'
#WRITE_OUTPUT_PATH = './data/output.txt'
EPOCH= 20
BATCH_SIZE=  8
IMAGE_SIZE = (224,224)
LAYER_N = 3
DROPOUT= 0.5
LEARNING_RATE = 1E-5
SAVE_TRAIN_PATH = ('./data/train_x.npy', 'data/train_y.npy')
SAVE_TEST_PATH = './data/test_x.npy'
VOLCABULARY_PATH = './vocab.txt'
TENSORBOARD_DIR = './runs'


dm = Datamanager(vocabulary_file = VOLCABULARY_PATH, tensorboard_dir = TENSORBOARD_DIR)
train_dataloader = dm.get_data_movie('./data/train/movie/','./data/train/tag.txt',batch_size=BATCH_SIZE, shuffle = True, image_size = IMAGE_SIZE, save_path = SAVE_TRAIN_PATH)
#test_dataloader = dm.get_test_data_movie('./data/test/movie/',batch_size=BATCH_SIZE, shuffle = True, image_size = IMAGE_SIZE, save_path = SAVE_TEST_PATH)


model= Movie_Classifier(IMAGE_SIZE, layer_n  = LAYER_N, label_dim = dm.voc.n_words, dropout = DROPOUT).cuda()
print('Model parameters: {}'.format(dm.count_parameters(model)))

optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

accu_record=0
for epoch in range(1,EPOCH+1):
    dm.train_movie( model, train_dataloader, epoch, optimizer)
    #record=dm.val_classifier( model, val_dataloader, epoch)
    '''
    if record[1]> accu_record:
        torch.save(model,OUTPUT_PATH)
        accu_record= record[1]
        print('Model saved!!!')
    '''
    print('='*80)
#result = dm.test_rnn( model, test_dataloader)
