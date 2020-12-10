import torch
from torch.utils.data import DataLoader
from transformer import Transformer, Config
from data.dataset import CorpusDataset, TokenSentenceConverter
import tqdm
import time
from evaluation.translate import translate_batch
from config import *

model = Transformer(Config(model_config))
model.load_state_dict(torch.load('model_state_dict/5epoch/transformer.pkl'))
model.cuda()
model.eval()
batch_size = 32
converter = TokenSentenceConverter('data/vocab.pkl')
dataset = CorpusDataset('data/corpus/valid_en', 'data/corpus/valid_cn', converter, to_token=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda x: ([s[0] for s in x], [s[1] for s in x]))
translate = lambda x: translate_batch(model, converter, [x])
bleu_ = 0
dataloader = iter(dataloader)
batches = 10
with torch.no_grad():
    for _ in tqdm.tqdm(range(batches)):
        src, tgt = next(dataloader)
        results = translate_batch(model, converter, src, tgt)
        for _, _, _, b in results:
            bleu_ += b
print(bleu_ / batch_size / batches)
