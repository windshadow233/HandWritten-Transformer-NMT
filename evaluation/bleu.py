from evaluation.beam_search import *
from data.dataset import CorpusDataset

dataset = CorpusDataset('../data/corpus/train_en', '../data/corpus/train_cn', '../data/vocab.pkl', model_config)


def bleu(src, targets):
    pass


def translate(sentences):
    """
    :param sentences: 一个英文句子列表
    :return: results: 中文句子列表
    """
    token = torch.nn.utils.rnn.pad_sequence([dataset.sentence2token(sen, 'en').to(device)
                                             for sen in sentences],
                                            batch_first=True)
    decode = beam_search(token, max_length=100, num_beams=3)
    results = [dataset.token2sentence(sen, 'cn') for sen in decode]
    return results
