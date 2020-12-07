from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
from transformer import Transformer, Config
from data.dataset import TokenSentenceConverter, CorpusDataset, sentence_collate_fn
from config import *


class Trainer(object):
    def __init__(self, model, model_config, train_set,
                 lr=1e-4, model_state_dict=None,
                 optimizer_state_dict=None, use_cuda=True, seed=10):
        self.seed = seed
        # 初始化之前固定种子
        torch.manual_seed(seed)
        self.device = torch.device('cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.model = model(Config(model_config))
        self.model: Transformer
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        self.train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=sentence_collate_fn)
        self.optimizer = Adam(self.model.parameters(), lr, betas=(0.9, 0.98), eps=1e-9)
        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)
        self.loss_fcn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def train(self, epochs):
        for epoch in range(epochs):
            for i, (src, tgt) in tqdm.tqdm(enumerate(self.train_loader), desc='Epoch_%s' % epoch, total=len(self.train_loader)):
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                output, gold = self.model(src, tgt)
                loss = self.loss_fcn(output.transpose(1, 2), gold)
                print('Loss: ', loss.item())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()


if __name__ == '__main__':
    converter = TokenSentenceConverter('data/vocab.pkl')
    train_set = CorpusDataset('data/corpus/train_en', 'data/corpus/train_cn', converter)
    trainer = Trainer(
        model=Transformer,
        model_config=model_config,
        train_set=train_set,
        lr=2e-4,
        # model_state_dict=torch.load('model/3epoch/transformer.pkl'),
        # optimizer_state_dict=torch.load('model/3epoch/optimizer.pkl'),
        seed=10
    )
    trainer.train(5)
    torch.save(trainer.model.state_dict(), 'model/1epoch/transformer.pkl')
    torch.save(trainer.optimizer.state_dict(), 'model/1epoch/optimizer.pkl')
