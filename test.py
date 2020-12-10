import torch
from torch.utils.data import DataLoader
from transformer import Transformer, Config
from data.dataset import CorpusDataset, TokenSentenceConverter, sentence_collate_fn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import tqdm
import time
from evaluation.translate import translate_batch
from config import *

model = Transformer(Config(model_config))
model.load_state_dict(torch.load('model_state_dict/5epoch/transformer.pkl'))
model.cuda()
model.eval()
converter = TokenSentenceConverter('data/vocab.pkl')
dataset = CorpusDataset('data/corpus/valid_en', 'data/corpus/valid_cn', converter, to_token=False)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False,
                        collate_fn=lambda x: ([s[0] for s in x], [s[1] for s in x]))
translate = lambda x: list(translate_batch(model, converter, [x]).values())[0]
bleu_ = 0
fcn = SmoothingFunction()
dataloader = iter(dataloader)
# with torch.no_grad():
#     for _ in tqdm.tqdm(range(30)):
#         src, tgt = next(dataloader)
#         src = [converter.token2sentence(sen, 'en') for sen in src]
#         tgt = [converter.token2sentence(sen, 'cn').replace('<sos>', '').replace('<eos>', '') for sen in tgt]
#         result = translate_batch(model, converter, src).values()
#         for r, t in zip(result, tgt):
#             bleu_ += sentence_bleu([list(t)], list(r), weights=(0.25,) * 4, smoothing_function=fcn.method1)
# print(bleu_ / 960)
