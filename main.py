import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
from transformer.transformer import Transformer, Config
from data.dataset import CorpusDataset
from configparser import ConfigParser


parser = ConfigParser()
parser.read('./config.ini', encoding='utf-8')
model_config = parser['model_config']


def sentence_collate_fn(one_batch):
    src = [sen[0] for sen in one_batch]
    tgt = [sen[1] for sen in one_batch]
    src = nn.utils.rnn.pad_sequence(src, batch_first=True)
    tgt = nn.utils.rnn.pad_sequence(tgt, batch_first=True)
    return src, tgt


torch.manual_seed(10)
epochs = 2
device = torch.device('cuda:0')
transform = Transformer(Config(model_config))
transform.to(device)
dataset = CorpusDataset('data/corpus/train_en', 'data/corpus/train_cn', 'data/vocab.pkl', model_config)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=sentence_collate_fn)
optimizer = Adam(transform.parameters(), lr=2e-5)
loss_fcn = nn.CrossEntropyLoss(ignore_index=0)
for epoch in range(epochs):
    for src, tgt in tqdm.tqdm(dataloader, desc='Epoch_%s' % epoch):
        src = src.to(device)
        tgt = tgt.to(device)
        output, label = transform(src, tgt)
        loss = loss_fcn(output.transpose(1, 2), label)
        print('Loss: ', loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
