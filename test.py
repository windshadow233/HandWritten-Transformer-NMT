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
bleu1 = bleu2 = bleu3 = bleu4 = 0
dataloader = iter(dataloader)
batches = 2
translate_result = 'Top 50 Results:\n\n'
with torch.no_grad(), tqdm.tqdm(range(batches)) as t:
    for _ in t:
        src, tgt = next(dataloader)
        result, s = translate_batch(model, converter, src, tgt)
        translate_result += s
        bleu1 += sum(result.get('bleu1'))
        bleu2 += sum(result.get('bleu2'))
        bleu3 += sum(result.get('bleu3'))
        bleu4 += sum(result.get('bleu4'))
translate_result += f'Top 50 BLEU1: {bleu1 / batches / batch_size} | BLEU2: {bleu2 / batches / batch_size} | ' \
                    f'BLEU3: {bleu3 / batches / batch_size} | BLEU4: {bleu4 / batches / batch_size}\n'
with open('evaluation/test_results.txt', 'w', encoding='utf-8') as f:
    f.write(translate_result)


batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda x: ([s[0] for s in x], [s[1] for s in x]))
bleu1 = bleu2 = bleu3 = bleu4 = 0
count = 0
with torch.no_grad(), tqdm.tqdm(dataloader) as t:
    for src, tgt in t:
        result, s = translate_batch(model, converter, src, tgt)
        count += len(tgt)
        bleu1 += sum(result.get('bleu1'))
        bleu2 += sum(result.get('bleu2'))
        bleu3 += sum(result.get('bleu3'))
        bleu4 += sum(result.get('bleu4'))
        print(f'BLEU1: {bleu1 / count}\nBLEU2: {bleu2 / count}\nBLEU3: {bleu3 / count}\nBLEU4: {bleu4 / count}')
with open('evaluation/test_results.txt', 'a+', encoding='utf-8') as f:
    f.write(f'\nTotal BLEU1: {bleu1 / len(dataloader.dataset)} | BLEU2: {bleu2 / len(dataloader.dataset)} | '
            f'BLEU3: {bleu3 / len(dataloader.dataset)} | BLEU4: {bleu4 / len(dataloader.dataset)}')
