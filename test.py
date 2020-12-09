import torch
from torch.utils.data import DataLoader
from transformer import Transformer, Config
from data.dataset import CorpusDataset, TokenSentenceConverter, sentence_collate_fn
from nltk.translate.bleu_score import sentence_bleu
from nltk import bleu
import tqdm
import time
from evaluation.translate import translate_batch
from config import *

model = Transformer(Config(model_config))
model.load_state_dict(torch.load('model_state_dict/1epoch/transformer.pkl'))
model.cuda()
model.eval()
converter = TokenSentenceConverter('data/vocab.pkl')
dataset = CorpusDataset('data/corpus/test_en', 'data/corpus/test_cn', converter)
dataloader = DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=sentence_collate_fn)
translate = lambda x: list(translate_batch(model, converter, [x]).items())[0][1]
bleu_ = 0
dataloader = iter(dataloader)
# with torch.no_grad():
#     for _ in tqdm.tqdm(range(50)):
#         src, tgt = next(dataloader)
#         src = [converter.token2sentence(sen, 'en') for sen in src]
#         tgt = [converter.token2sentence(sen[1: -1], 'cn') for sen in tgt]
#         result = translate(model, converter, src)
#         for r, t in zip(result, tgt):
#             bleu_ += bleu([list(t)], list(r))
# print(bleu_ / 1000)
