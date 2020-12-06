import torch
from torch.utils.data import DataLoader
from transformer import Transformer, Config
from data.dataset import CorpusDataset, TokenSentenceConverter, sentence_collate_fn
from nltk.translate.bleu_score import sentence_bleu
from nltk import bleu
from evaluation.translate import translate
from config import *

model = Transformer(Config(model_config))
model.load_state_dict(torch.load('model/3epoch/transformer.pkl'))
model.cuda()
model.eval()
converter = TokenSentenceConverter('data/vocab.pkl')
dataset = CorpusDataset('data/corpus/test_en', 'data/corpus/test_cn', converter)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=sentence_collate_fn)
translate_one_sen = lambda x: translate(model, converter, [x])
with torch.no_grad():
    dataloader = iter(dataloader)
    src, tgt = next(dataloader)
    src = [converter.token2sentence(sen, 'en') for sen in src]
    tgt = [converter.token2sentence(sen, 'cn').replace('<sos>', '').replace('<eos>', '') for sen in tgt]
    result = translate(model, converter, src)
