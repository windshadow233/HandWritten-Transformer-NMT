from evaluation.beam_search import *


def bleu(src, targets):
    pass


def translate(model, converter, sentences):
    """
    :param model: Transformer
    :param converter: TokenSentenceConverter
    :param sentences: 一个英文句子列表
    :return: results: 中文句子列表
    """
    token = torch.nn.utils.rnn.pad_sequence([converter.sentence2token(sen, 'en').to(device)
                                             for sen in sentences],
                                            batch_first=True)
    decode = beam_search(model, token, max_length=100, num_beams=3)
    results = [converter.token2sentence(sen, 'cn') for sen in decode]
    return results
