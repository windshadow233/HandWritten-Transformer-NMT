from collections import Counter
from tqdm import tqdm
import nltk
import pickle
from data_utils import *
from config import *


def process(file, lang='zh'):
    print('processing {}...'.format(file))
    with open(file, 'r', encoding='utf-8') as f:
        data = f.readlines()

    word_freq = Counter()

    for line in tqdm(data):
        sentence = line.strip()
        if lang == 'en':
            sentence_en = sentence.lower()
            tokens = [normalize_string(s) for s in nltk.word_tokenize(sentence_en)]
            word_freq.update(tokens)
            vocab_size = n_src_vocab
        else:
            tokens = list(sentence)
            word_freq.update(tokens)
            vocab_size = n_tgt_vocab
    print(len(word_freq))
    words = word_freq.most_common(vocab_size - 4)
    word_map = {k[0]: v + 4 for v, k in enumerate(words)}
    word_map['<pad>'] = pad_idx
    word_map['<sos>'] = sos_idx
    word_map['<eos>'] = eos_idx
    word_map['<unk>'] = unk_idx

    word2idx = word_map
    idx2word = {v: k for k, v in word2idx.items()}

    return word2idx, idx2word


if __name__ == '__main__':

    vocab_file = 'vocab.pkl'
    n_src_vocab = n_tgt_vocab = vocab_size
    train_translation_en_filename = 'corpus/train_en'
    train_translation_zh_filename = 'corpus/train_cn'
    valid_translation_en_filename = 'corpus/valid_en'
    valid_translation_zh_filename = 'corpus/valid_cn'

    tgt_char2idx, tgt_idx2char = process(train_translation_zh_filename, lang='zh')
    src_char2idx, src_idx2char = process(train_translation_en_filename, lang='en')
    data = {
        'vocab_dict': {
            'src_char2idx': src_char2idx,
            'src_idx2char': src_idx2char,
            'tgt_char2idx': tgt_char2idx,
            'tgt_idx2char': tgt_idx2char
        }
    }
    with open(vocab_file, 'wb') as file:
        pickle.dump(data, file)
