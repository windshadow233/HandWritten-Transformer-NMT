from torch import nn
from torch.utils.data import Dataset
import pickle
import nltk
from .utils import *
from config import *


def sentence_collate_fn(one_batch):
    src = [sen[0] for sen in one_batch]
    tgt = [sen[1] for sen in one_batch]
    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=pad_idx)
    tgt = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=pad_idx)
    return src, tgt


class TokenSentenceConverter(object):
    def __init__(self, vocab_dict):
        with open(vocab_dict, 'rb') as f:
            vocab_dict = pickle.load(f).get('vocab_dict')
        self.src_char2idx = vocab_dict.get('src_char2idx')
        self.src_idx2char = vocab_dict.get('src_idx2char')
        self.tgt_char2idx = vocab_dict.get('tgt_char2idx')
        self.tgt_idx2char = vocab_dict.get('tgt_idx2char')

    def sentence2token(self, sentence, lang='en'):
        assert lang in {'en', 'cn'}
        sentence = full_width2half_width(sentence)
        char2idx = self.src_char2idx if lang == 'en' else self.tgt_char2idx
        if lang == 'en':
            sentence = [normalize_string(s.strip()) for s in nltk.word_tokenize(sentence)]
            sentence = sentence[:max_seq_len]
        else:
            sentence = sentence[:max_seq_len - 2]
            sentence = ['<sos>'] + list(sentence) + ['<eos>']
        return torch.tensor([char2idx.get(word, char2idx['<unk>']) for word in sentence], dtype=torch.long)

    def token2sentence(self, token, lang='en'):
        assert lang in {'en', 'cn'}
        if isinstance(token, torch.Tensor):
            token = token.tolist()
        idx2char = self.src_idx2char if lang == 'en' else self.tgt_idx2char
        sentence = [idx2char.get(idx) for idx in token if idx != pad_idx]
        if lang == 'en':
            return ' '.join(sentence)
        return ''.join(sentence)


class CorpusDataset(Dataset):
    def __init__(self, src_file, tgt_file, converter: TokenSentenceConverter):
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_sentences = f.readlines()
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_sentences = f.readlines()
        self.converter = converter

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, item, to_token=True):
        src_sen = self.src_sentences[item].strip()
        tgt_sen = self.tgt_sentences[item].strip()
        if to_token:
            src_token = self.converter.sentence2token(src_sen, 'en')
            tgt_token = self.converter.sentence2token(tgt_sen, 'cn')
            return src_token, tgt_token
        return src_sen, tgt_sen
