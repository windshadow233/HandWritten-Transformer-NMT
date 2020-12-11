from torch.utils.data import DataLoader
from transformer import Transformer, Config
from data.dataset import CorpusDataset, TokenSentenceConverter
import tqdm
from evaluation.translate import translate_batch
from config import *

model = Transformer(Config(model_config))
model.load_state_dict(torch.load('model_state_dict/5epoch/transformer.pkl'))
model.cuda()
model.eval()
batch_size = 25
converter = TokenSentenceConverter('data/vocab.pkl')
dataset = CorpusDataset('data/corpus/test_en', 'data/corpus/test_cn', converter, to_token=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda x: ([s[0] for s in x], [s[1] for s in x]))
translate = lambda x: translate_batch(model, converter, [x])[1]
bleu_ = 0
dataloader = iter(dataloader)
batches = 2
translate_result = 'Top 50 Results:\n\n'
with torch.no_grad(), tqdm.tqdm(range(batches)) as t:
    for _ in t:
        src, tgt = next(dataloader)
        result, s = translate_batch(model, converter, src, tgt)
        translate_result += s
        bleu_ += sum(map(lambda x: x[-1], result))
translate_result += f'Top 50 BLEU: {bleu_ / batches / batch_size}'
with open('data/results/test_results.txt', 'w', encoding='utf-8') as f:
    f.write(translate_result)


batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda x: ([s[0] for s in x], [s[1] for s in x]))
bleu_ = 0
count = 0
with torch.no_grad(), tqdm.tqdm(dataloader) as t:
    for src, tgt in t:
        result, s = translate_batch(model, converter, src, tgt)
        count += len(tgt)
        bleu_ += sum(map(lambda x: x[-1], result))
        print(f'BLEU: {bleu_ / count}')
with open('data/results/test_results.txt', 'a+', encoding='utf-8') as f:
    f.write(f'\nTotal BLEU: {bleu_ / len(dataloader.dataset)}')
