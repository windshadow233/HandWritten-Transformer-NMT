import torch
from torch.utils.data import Dataset
import pickle
import nltk
from data_utils import *


class CorpusDataset(Dataset):
    def __init__(self, src_file, tgt_file, vocab_dict, model_config):
        self.max_len = model_config.getint('max_seq_len')
        with open(vocab_dict, 'rb') as f:
            vocab_dict = pickle.load(f).get('vocab_dict')
        self.src_char2idx = vocab_dict.get('src_char2idx')
        self.src_idx2char = vocab_dict.get('src_idx2char')
        self.tgt_char2idx = vocab_dict.get('tgt_char2idx')
        self.tgt_idx2char = vocab_dict.get('tgt_idx2char')
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_sentences = f.readlines()
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_sentences = f.readlines()

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, item):
        src_sen = self.src_sentences[item].strip()
        tgt_sen = self.tgt_sentences[item].strip()
        src_token = self.sentence2token(src_sen, 'en')
        tgt_token = self.sentence2token(tgt_sen, 'zh')
        return src_token, tgt_token

    def sentence2token(self, sentence, lang='en'):
        assert lang in {'en', 'zh'}
        char2idx = self.src_char2idx if lang == 'en' else self.tgt_char2idx
        if lang == 'en':
            sentence = [normalize_string(s.strip()) for s in nltk.word_tokenize(sentence)]
        else:
            sentence = ['<sos>'] + list(sentence) + ['<eos>']
        sentence = sentence[:self.max_len]
        return torch.tensor([char2idx.get(word, char2idx['<unk>']) for word in sentence], dtype=torch.long)

    def token2sentence(self, token, lang='en'):
        assert lang in {'en', 'zh'}
        idx2char = self.src_idx2char if lang == 'en' else self.tgt_idx2char
        return [idx2char.get(idx) for idx in token]
