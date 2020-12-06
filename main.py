from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
from transformer.transformer import Transformer, Config
from data.dataset import CorpusDataset, sentence_collate_fn
from config import *


class Trainer(object):
    def __init__(self, model, model_config, train_set,
                 lr=1e-4, model_state_dict=None,
                 optimizer_state_dict=None, use_cuda=True, seed=10):
        self.seed = seed
        torch.manual_seed(seed)
        self.device = torch.device('cuda:0' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.model = model(Config(model_config))
        self.model: Transformer
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        self.train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=sentence_collate_fn)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(optimizer_state_dict)
        self.pad_idx = pad_idx
        self.loss_fcn = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    @torch.no_grad()
    def calculate_loss(self, dataloader):
        loss = 0
        for src, tgt in tqdm.tqdm(dataloader, desc='calculate loss', total=len(dataloader)):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            output, label = self.model(src, tgt)
            loss += self.loss_fcn(output.transpose(1, 2), label).item() * src.shape[0]
        return loss / len(dataloader.dataset)

    def train(self, epochs):
        torch.manual_seed(self.seed)
        for epoch in range(epochs):
            for i, (src, tgt) in tqdm.tqdm(enumerate(self.train_loader), desc='Epoch_%s' % epoch, total=len(self.train_loader)):
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                output, label = self.model(src, tgt)
                loss = self.loss_fcn(output.transpose(1, 2), label)
                print('Loss: ', loss.item())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()


if __name__ == '__main__':
    train_set = CorpusDataset('data/corpus/train_en', 'data/corpus/train_cn', 'data/vocab.pkl', model_config)
    trainer = Trainer(
        model=Transformer,
        model_config=model_config,
        train_set=train_set,
        lr=1e-4,
        seed=10
    )
    trainer.train(2)
