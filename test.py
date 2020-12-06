import torch
from torch.utils.data import DataLoader
from transformer import Transformer, Config
from data.dataset import CorpusDataset, TokenSentenceConverter, sentence_collate_fn
from evaluation.bleu import translate
from config import *

model = Transformer(Config(model_config))
model.load_state_dict(torch.load('model/transformer.pkl'))
model.cuda()
model.eval()
converter = TokenSentenceConverter('data/vocab.pkl')
dataset = CorpusDataset('data/corpus/test_en', 'data/corpus/test_cn', converter)
dataloader = DataLoader(dataset, batch_size=16, collate_fn=sentence_collate_fn)

