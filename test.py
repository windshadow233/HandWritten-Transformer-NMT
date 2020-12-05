import torch
from configparser import ConfigParser
from transformer.transformer import Transformer, Config
from data.dataset import CorpusDataset

parser = ConfigParser()
parser.read('./config.ini', encoding='utf-8')
model_config = parser['model_config']
pad_value = model_config.getint('pad_idx')
dataset = CorpusDataset('data/corpus/train_en', 'data/corpus/train_cn', 'data/vocab.pkl', model_config)
device = torch.device('cuda:0')
transform = Transformer(Config(model_config))
transform.load_state_dict(torch.load('transformer.pkl'))
transform.to(device)
transform.eval()


def translate(sen):
    token = dataset.sentence2token(sen, 'en').to(device)
    target = transform.beam_search(token, max_len=min(2 * len(token) + 2, model_config.getint('max_seq_len')))
    for t in target:
        result = dataset.token2sentence(t[0].squeeze(0), 'cn')
        print(result)
