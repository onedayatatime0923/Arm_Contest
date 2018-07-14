
from util import Datamanager, EncoderRNN, AttnDecoderRNN
import torch
assert torch and EncoderRNN and AttnDecoderRNN

EPOCHS= 50
BATCH_SIZE= 128
HIDDEN_LAYER= 1024
LAYER_N= 3
HOP_N= 3
DROPOUT= 0.5
MIN_COUNT = 3
PRINT_OUTPUT_PATH = './record.png'
WRITE_OUTPUT_PATH = './data/output.txt'
VOLCABULARY_PATH = './vocab.txt'


dm = Datamanager(min_count= MIN_COUNT)
dm.get_data('train','./data/training_data/feat','./data/training_label.json','train',batch_size=BATCH_SIZE)
dm.get_data('test','./data/testing_data/feat','./data/testing_label.json','test',batch_size=BATCH_SIZE,shuffle=False)
dm.voc.save(VOLCABULARY_PATH)
print('Max length: {}'.format(dm.max_length))
print('Vocabulary size: {}'.format(dm.voc.n_words))

encoder=EncoderRNN(4096,HIDDEN_LAYER, LAYER_N,dropout=DROPOUT).cuda()
decoder=AttnDecoderRNN(HIDDEN_LAYER,dm.vocab_size, LAYER_N, HOP_N, dropout=DROPOUT).cuda()
print("Encoder Parameter: {}".format(dm.count_parameters(encoder)))
print("Decoder Parameter: {}".format(dm.count_parameters(decoder)))
#torch.save(encoder,'encoder.pt')
#torch.save(decoder,'decoder.pt')

dm.trainIters(encoder, decoder, 'train', 'test', EPOCHS, write_file= WRITE_OUTPUT_PATH, plot_file = PRINT_OUTPUT_PATH)
#torch.save(encoder,'encoder.pt')
#torch.save(decoder,'decoder.pt')

