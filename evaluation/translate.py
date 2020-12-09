from evaluation.beam_search import *


@torch.no_grad()
def translate(model, converter, sentences):
    """
    :param model: Transformer
    :param converter: TokenSentenceConverter
    :param sentences: 一个英文句子列表
    :return: results: 中文句子列表
    """
    sentences = [s + '.' if not s.endswith(('.', '!', '?')) else s for s in sentences]
    token = torch.nn.utils.rnn.pad_sequence([converter.sentence2token(sen, 'en').to(device)
                                             for sen in sentences],
                                            batch_first=True)
    decode = beam_search(model, token, num_beams=5)
    results = [converter.token2sentence(sen, 'cn').replace('<sos>', '').replace('<eos>', '') for sen in decode]
    results = dict(zip(sentences, results))
    return results
